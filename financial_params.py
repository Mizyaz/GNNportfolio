import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

@dataclass
class FinancialParameters:
    """Configuration parameters for financial metrics calculation"""
    # Trend parameters
    ma_windows: List[int] = None  # Moving average windows
    momentum_windows: List[int] = None  # Momentum calculation windows
    volatility_windows: List[int] = None  # Volatility calculation windows
    
    # Diversification parameters
    min_weight: float = 0.0  # Minimum weight constraint
    max_weight: float = 1.0  # Maximum weight constraint
    risk_free_rate: float = 0.02  # Risk-free rate for Sharpe ratio
    
    # Technical parameters
    rsi_window: int = 14  # RSI calculation window
    bollinger_window: int = 20  # Bollinger Bands window
    bollinger_std: float = 2.0  # Number of standard deviations for Bollinger Bands
    
    def __post_init__(self):
        """Initialize default values if None"""
        if self.ma_windows is None:
            self.ma_windows = [5, 10, 20, 50]
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50]

class FinancialMetrics:
    def __init__(self, params: FinancialParameters = None):
        """Initialize with parameters"""
        self.params = params or FinancialParameters()
    
    def calculate_trend_metrics(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate trend-following metrics
        
        Parameters:
        -----------
        prices : np.ndarray
            Price matrix (time x assets)
            
        Returns:
        --------
        Dict with keys:
        - ma_ratios: Current price to moving average ratios
        - momentum: Price momentum over different windows
        - trend_strength: Trend strength indicators
        
        Expected ranges:
        - ma_ratios: typically 0.8-1.2
        - momentum: typically -0.2 to 0.2
        - trend_strength: 0 to 1
        """
        results = {}
        
        # Calculate moving average ratios
        ma_ratios = []
        for window in self.params.ma_windows:
            ma = self._calculate_ma(prices, window)
            ratio = prices[-1] / ma[-1]
            ma_ratios.append(ratio)
        results['ma_ratios'] = np.array(ma_ratios)
        
        # Calculate momentum
        momentum = []
        for window in self.params.momentum_windows:
            mom = (prices[-1] - prices[-window-1]) / prices[-window-1]
            momentum.append(mom)
        results['momentum'] = np.array(momentum)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(prices)
        results['trend_strength'] = trend_strength
        
        return results
    
    def calculate_diversification_metrics(self, returns: np.ndarray, 
                                       weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate diversification metrics
        
        Parameters:
        -----------
        returns : np.ndarray
            Returns matrix (time x assets)
        weights : np.ndarray, optional
            Portfolio weights (default: equal weights)
            
        Returns:
        --------
        Dict with keys:
        - herfindahl: Portfolio concentration (0-1)
        - diversification_ratio: Portfolio diversification measure
        - effective_n: Effective number of assets
        - correlation_penalty: Correlation-based penalty term
        
        Expected ranges:
        - herfindahl: 0 (perfect diversification) to 1 (concentration)
        - diversification_ratio: > 1 (higher is more diversified)
        - effective_n: 1 to number of assets
        - correlation_penalty: 0 (uncorrelated) to 1 (perfectly correlated)
        """
        if weights is None:
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            
        results = {}
        
        # Herfindahl Index (concentration measure)
        results['herfindahl'] = np.sum(weights ** 2)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns.T)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Individual asset volatilities
        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vols = weights * asset_vols
        
        # Diversification Ratio
        results['diversification_ratio'] = np.sum(weighted_vols) / portfolio_vol
        
        # Effective N
        results['effective_n'] = 1 / results['herfindahl']
        
        # Correlation penalty
        corr_matrix = np.corrcoef(returns.T)
        results['correlation_penalty'] = np.sum(weights @ corr_matrix @ weights)
        
        return results
    
    def calculate_technical_indicators(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate technical indicators
        
        Parameters:
        -----------
        prices : np.ndarray
            Price matrix (time x assets)
            
        Returns:
        --------
        Dict with keys:
        - rsi: Relative Strength Index
        - bb_position: Position within Bollinger Bands
        - volatility: Historical volatility measures
        
        Expected ranges:
        - rsi: 0 to 100
        - bb_position: -1 to 1 (position within bands)
        - volatility: typically 0.01 to 0.5 (annualized)
        """
        results = {}
        
        # Calculate RSI
        results['rsi'] = self._calculate_rsi(prices)
        
        # Calculate Bollinger Bands position
        results['bb_position'] = self._calculate_bb_position(prices)
        
        # Calculate volatility measures
        vol_measures = []
        for window in self.params.volatility_windows:
            returns = np.log(prices[1:] / prices[:-1])
            vol = np.std(returns[-window:], axis=0) * np.sqrt(252)  # Annualized
            vol_measures.append(vol)
        results['volatility'] = np.array(vol_measures)
        
        return results
    
    def _calculate_ma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        return np.array([np.mean(prices[max(0, i-window+1):i+1], axis=0) 
                        for i in range(len(prices))])
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> np.ndarray:
        """Calculate trend strength using linear regression R²"""
        x = np.arange(len(prices)).reshape(-1, 1)
        trend_strength = np.array([
            stats.linregress(x.flatten(), prices[:, i]).rvalue ** 2
            for i in range(prices.shape[1])
        ])
        return trend_strength
    
    def _calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """Calculate RSI"""
        returns = np.diff(prices, axis=0)
        gains = np.maximum(returns, 0)
        losses = -np.minimum(returns, 0)
        
        avg_gains = np.mean(gains[-self.params.rsi_window:], axis=0)
        avg_losses = np.mean(losses[-self.params.rsi_window:], axis=0)
        
        rs = avg_gains / (avg_losses + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bb_position(self, prices: np.ndarray) -> np.ndarray:
        """Calculate position within Bollinger Bands"""
        ma = self._calculate_ma(prices, self.params.bollinger_window)
        std = np.std(prices[-self.params.bollinger_window:], axis=0)
        
        upper = ma[-1] + self.params.bollinger_std * std
        lower = ma[-1] - self.params.bollinger_std * std
        
        position = (prices[-1] - ma[-1]) / (upper - lower + 1e-6)
        return np.clip(position, -1, 1) 