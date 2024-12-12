# metrics_computer.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class MetricsConfig:
    risk_free_rate: float = 0.02  # Annual risk-free rate
    benchmark_returns: np.ndarray = None  # Optional benchmark returns for Alpha and Beta

class MetricsComputer:
    """
    Computes various financial metrics based on portfolio returns and weights.
    """
    def __init__(self, config: MetricsConfig = MetricsConfig()):
        self.risk_free_rate = config.risk_free_rate
        self.benchmark_returns = config.benchmark_returns  # Should be aligned with portfolio returns
    
    def compute(self, current_weights: np.ndarray, observation: Dict[str, np.ndarray],
               current_step_returns: np.ndarray, metrics_list: List[Tuple[str, float]]) -> float:
        """
        Compute the weighted sum of specified metrics.
        
        Args:
            current_weights (np.ndarray): Current portfolio weights.
            observation (Dict[str, np.ndarray]): Observation containing past returns.
                Expected keys: 'returns' (window_size x num_assets)
            current_step_returns (np.ndarray): Returns of the current step.
            metrics_list (List[Tuple[str, float]]): List of (metric_name, weight) tuples.
        
        Returns:
            float: The weighted sum of computed metrics as reward.
        """
        computed_metrics = {}
        for metric_name, weight in metrics_list:
            func = getattr(self, f"calculate_{metric_name.lower()}", None)
            if func:
                try:
                    metric_value = func(current_weights, observation, current_step_returns)
                    computed_metrics[metric_name] = metric_value * weight
                except Exception as e:
                    print(f"Error computing {metric_name}: {e}")
                    computed_metrics[metric_name] = 0.0
            else:
                print(f"Metric {metric_name} not recognized.")
                computed_metrics[metric_name] = 0.0
        
        # Sum all weighted metrics
        total_reward = sum(computed_metrics.values())
        return total_reward
    
    def calculate_sharpe_ratio(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                               current_step_returns: np.ndarray) -> float:
        """
        Calculate the Sharpe Ratio.
        
        Returns:
            float: Sharpe Ratio.
        """
        # Calculate portfolio returns over the window
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        mean_return = np.mean(portfolio_returns) * 252  # Annualize
        std_return = np.std(portfolio_returns) * np.sqrt(252)  # Annualize
        
        sharpe = (mean_return - self.risk_free_rate) / (std_return + 1e-6)
        return sharpe
    
    def calculate_sortino_ratio(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                current_step_returns: np.ndarray) -> float:
        """
        Calculate the Sortino Ratio.
        
        Returns:
            float: Sortino Ratio.
        """
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        mean_return = np.mean(portfolio_returns) * 252  # Annualize
        
        # Downside deviation
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
        
        sortino = (mean_return - self.risk_free_rate) / (downside_std + 1e-6)
        return sortino
    
    def calculate_maximum_drawdown(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                   current_step_returns: np.ndarray) -> float:
        """
        Calculate the Maximum Drawdown.
        
        Returns:
            float: Maximum Drawdown (as a positive number).
        """
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdowns)
        return max_drawdown
    
    def calculate_cumulative_returns(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                    current_step_returns: np.ndarray) -> float:
        """
        Calculate the Cumulative Returns.
        
        Returns:
            float: Cumulative Return.
        """
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        return cumulative_return
    
    def calculate_portfolio_volatility(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                       current_step_returns: np.ndarray) -> float:
        """
        Calculate the Portfolio Volatility.
        
        Returns:
            float: Portfolio Volatility (annualized).
        """
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualize
        return volatility
    
    def calculate_alpha(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                        current_step_returns: np.ndarray) -> float:
        """
        Calculate Alpha: Portfolio's excess return over the benchmark.
        
        Returns:
            float: Alpha.
        """
        if self.benchmark_returns is None:
            # If no benchmark is provided, return 0
            return 0.0
        
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        benchmark_returns = self.benchmark_returns[:len(portfolio_returns)]
        
        # Calculate mean returns
        portfolio_mean = np.mean(portfolio_returns) * 252  # Annualize
        benchmark_mean = np.mean(benchmark_returns) * 252  # Annualize
        
        alpha = portfolio_mean - benchmark_mean
        return alpha
    
    def calculate_beta(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                       current_step_returns: np.ndarray) -> float:
        """
        Calculate Beta: Portfolio's sensitivity to benchmark returns.
        
        Returns:
            float: Beta.
        """
        if self.benchmark_returns is None:
            # If no benchmark is provided, return 1
            return 1.0
        
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        benchmark_returns = self.benchmark_returns[:len(portfolio_returns)]
        
        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        beta = covariance / (benchmark_variance + 1e-6)
        return beta
    
    def calculate_turnover_rate(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                current_step_returns: np.ndarray) -> float:
        """
        Calculate the Turnover Rate: Sum of absolute changes in weights.
        
        Returns:
            float: Turnover Rate.
        """
        turnover = np.sum(np.abs(weights - self.previous_weights))
        return turnover
    
    def calculate_information_ratio(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                    current_step_returns: np.ndarray) -> float:
        """
        Calculate the Information Ratio.
        
        Returns:
            float: Information Ratio.
        """
        if self.benchmark_returns is None:
            return 0.0
        
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        benchmark_returns = self.benchmark_returns[:len(portfolio_returns)]
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        mean_active = np.mean(active_returns) * 252  # Annualize
        std_active = np.std(active_returns) * np.sqrt(252)  # Annualize
        
        information_ratio = mean_active / (std_active + 1e-6)
        return information_ratio
    
    def calculate_diversification_metrics(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                          current_step_returns: np.ndarray) -> float:
        """
        Calculate Diversification Metrics: Herfindahl-Hirschman Index (HHI) and Effective Number of Assets.
        
        Returns:
            float: Diversification Score (sum of normalized HHI and Effective N).
        """
        # Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(weights ** 2)
        
        # Effective Number of Assets
        effective_n = 1 / hhi if hhi > 0 else 0
        
        # Normalize HHI and Effective N
        normalized_hhi = hhi  # HHI ranges from 1/N to 1
        normalized_effective_n = effective_n / self.num_assets  # Effective N ranges from 1/N to 1
        
        diversification_score = normalized_hhi + normalized_effective_n
        return diversification_score
    
    def calculate_value_at_risk(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                current_step_returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Returns:
            float: VaR at the specified confidence level.
        """
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        return abs(var)
    
    def calculate_conditional_value_at_risk(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                           current_step_returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Returns:
            float: CVaR at the specified confidence level.
        """
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return abs(cvar)
    
    def calculate_transaction_costs(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                    current_step_returns: np.ndarray, transaction_cost_rate: float = 0.001) -> float:
        """
        Calculate Transaction Costs based on weight changes.
        
        Returns:
            float: Transaction Costs.
        """
        turnover = np.sum(np.abs(weights - self.previous_weights))
        transaction_costs = turnover * transaction_cost_rate
        return transaction_costs
    
    def calculate_liquidity_metrics(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                    current_step_returns: np.ndarray) -> float:
        """
        Calculate Liquidity Metrics.
        Placeholder implementation as liquidity data is not available.
        
        Returns:
            float: Liquidity Score (placeholder).
        """
        # Placeholder: Assume all assets have high liquidity
        return 1.0
    
    def calculate_exposure_metrics(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                   current_step_returns: np.ndarray) -> float:
        """
        Calculate Exposure Metrics.
        Placeholder implementation as sector/geography data is not available.
        
        Returns:
            float: Exposure Score (placeholder).
        """
        # Placeholder: Assume balanced exposure
        return 1.0
    
    def calculate_drawdown_duration(self, weights: np.ndarray, observation: Dict[str, np.ndarray],
                                    current_step_returns: np.ndarray) -> float:
        """
        Calculate Drawdown Duration: Time spent below previous peak.
        
        Returns:
            float: Average Drawdown Duration (days).
        """
        past_returns = observation['returns']  # (window_size, num_assets)
        portfolio_returns = np.dot(past_returns, weights)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        drawdowns = drawdowns > 0
        
        durations = []
        current_duration = 0
        for draw in drawdowns:
            if draw:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)
        average_duration = np.mean(durations) if durations else 0.0
        return average_duration
