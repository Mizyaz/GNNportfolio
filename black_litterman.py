import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from typing import Optional, Tuple, Union
import warnings

class BlackLittermanModel:
    def __init__(self, num_assets: int, window_size: int = 20, 
                 risk_free_rate: float = 0.02, market_return: float = 0.08,
                 risk_aversion: float = 2.0, tau: float = 0.05,
                 decay_factor: float = 0.94, regularization_factor: float = 1e-6,
                 use_exponential_weights: bool = True):
        """Enhanced initialization with additional parameters"""
        self.num_assets = num_assets
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.decay_factor = decay_factor
        self.regularization_factor = regularization_factor
        self.use_exponential_weights = use_exponential_weights

    def _prepare_returns(self, returns: np.ndarray) -> np.ndarray:
        """Prepare and validate returns data"""
        if returns.ndim == 1:
            returns = returns.reshape(-1, self.num_assets)
        elif returns.shape[1] != self.num_assets:
            raise ValueError(f"Expected {self.num_assets} assets, got {returns.shape[1]}")
        
        if len(returns) < 2:
            raise ValueError("Insufficient data points")
            
        return returns

    def _calculate_weights(self, length: int) -> np.ndarray:
        """Calculate time decay weights"""
        if self.use_exponential_weights:
            weights = np.power(self.decay_factor, np.arange(length-1, -1, -1))
            return weights / weights.sum()
        return np.ones(length) / length

    def _calculate_prior(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced prior calculation with weights"""
        try:
            returns = self._prepare_returns(returns)
            weights = self._calculate_weights(len(returns))
            
            # Calculate weighted covariance
            weighted_returns = returns * weights.reshape(-1, 1)
            weighted_mean = weighted_returns.sum(axis=0)
            centered_returns = returns - weighted_mean
            cov_matrix = (centered_returns.T @ (centered_returns * weights.reshape(-1, 1)))
            
            # Ensure positive definiteness
            cov_matrix = cov_matrix + np.eye(self.num_assets) * self.regularization_factor
            
            # Calculate market parameters
            market_premium = self.market_return - self.risk_free_rate
            mkt_weights = np.ones(self.num_assets) / self.num_assets
            
            # Calculate betas
            market_var = mkt_weights @ cov_matrix @ mkt_weights
            betas = (cov_matrix @ mkt_weights) / market_var
            
            # Calculate prior returns
            prior_returns = self.risk_free_rate + betas * market_premium
            
            return prior_returns, cov_matrix
            
        except Exception as e:
            warnings.warn(f"Prior calculation failed: {str(e)}")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get default parameters when calculations fail"""
        return (np.ones(self.num_assets) * self.risk_free_rate,
                np.eye(self.num_assets) * 0.01)

    def _incorporate_views(self, prior_returns, prior_cov, returns):
        """Incorporate investor views using recent performance"""
        try:
            # Use recent returns as views
            recent_returns = returns[-5:].mean(axis=0)
            
            # Confidence in views (using standard error)
            view_confidence = 1 / (returns[-5:].std(axis=0) + 1e-6)
            
            # Create view matrix (diagonal for asset-specific views)
            P = np.eye(self.num_assets)
            
            # Create view vector
            q = recent_returns
            
            # Create view uncertainty matrix
            omega = np.diag(1 / (view_confidence + 1e-6))
            
            # Calculate posterior parameters
            inv_prior_cov = np.linalg.inv(prior_cov)
            inv_omega = np.linalg.inv(omega)
            
            posterior_cov = np.linalg.inv(inv_prior_cov + np.dot(P.T, np.dot(inv_omega, P)))
            posterior_returns = posterior_cov @ (inv_prior_cov @ prior_returns + 
                                              np.dot(P.T, np.dot(inv_omega, q)))
            
            return posterior_returns, posterior_cov
            
        except Exception as e:
            print(f"Error in view incorporation: {e}")
            return prior_returns, prior_cov
    
    def optimize(self, returns_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize portfolio using Black-Litterman model with robust handling
        """
        try:
            # Ensure returns_data is numpy array with correct shape
            returns = np.asarray(returns_data)
            if len(returns) < 2:
                return self._get_default_parameters()
            
            # Add small noise to prevent singularity
            returns = returns + np.random.normal(0, 1e-8, returns.shape)
            
            # Calculate prior with regularization
            prior_returns, prior_cov = self._calculate_prior(returns)
            
            # Add regularization to covariance matrix
            prior_cov += np.eye(self.num_assets) * self.regularization_factor
            
            # Incorporate views with regularization
            posterior_returns, posterior_cov = self._incorporate_views(
                prior_returns, prior_cov, returns)
            
            return posterior_returns, posterior_cov
            
        except Exception as e:
            return self._get_default_parameters() 