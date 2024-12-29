# env/reward.py

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
import talib
from scipy import stats

from .risk_metrics import RiskMetricsComputer


@dataclass
class RewardConfig:
    """
    Configuration for the reward system.
    Each reward function is defined as a tuple:
    (function, weight, kwargs)
    """
    rewards: List[Tuple[Callable, float, Dict[str, Any]]] = field(default_factory=list)


class Reward:
    """
    Reward class to compute combined rewards based on multiple reward functions.
    Each reward is normalized to be within [-1, 1] using tanh scaling.
    """

    def __init__(self, config: RewardConfig):
        """
        Initialize the Reward class with a given configuration.

        Args:
            config (RewardConfig): Configuration containing reward functions, their weights, and kwargs.
        """
        self.rewards = config.rewards

    def compute_reward(self, env) -> float:
        """
        Compute the combined reward by iterating through all reward functions.
        Each reward is normalized to [-1, 1] using tanh(raw_reward / scaling_factor).

        Args:
            env (PortfolioEnv): The environment instance.

        Returns:
            float: The combined normalized reward.
        """
        total_reward = 0.0
        for func, weight, kwargs in self.rewards:
            raw_reward = func(env, **kwargs)
            scaling_factor = kwargs.get('scaling_factor', 1.0)
            # Prevent division by zero or extremely small scaling factors
            scaling_factor = max(scaling_factor, 1e-8)
            normalized_reward = np.tanh(raw_reward / scaling_factor)
            total_reward += weight * normalized_reward
        return total_reward


# Sample Reward Functions

def compute_sharpe(env, window: int = 20, risk_free_rate: float = 0.02, scaling_factor: float = 1.0) -> float:
    """
    Compute the rolling Sharpe ratio as a reward.

    Args:
        env (PortfolioEnv): The environment instance.
        window (int): Number of steps to consider for rolling computation.
        risk_free_rate (float): Annual risk-free rate.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Sharpe ratio.
    """
    start_idx = max(0, env.current_step - window)
    realized_returns = np.dot(env.returns.iloc[start_idx:env.current_step].values, env.previous_weights)
    sharpe = RiskMetricsComputer.compute_rolling_sharpe(realized_returns, risk_free_rate, window)
    return sharpe


def compute_sortino(env, window: int = 20, risk_free_rate: float = 0.02, scaling_factor: float = 1.0) -> float:
    """
    Compute the rolling Sortino ratio as a reward.

    Args:
        env (PortfolioEnv): The environment instance.
        window (int): Number of steps to consider for rolling computation.
        risk_free_rate (float): Annual risk-free rate.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Sortino ratio.
    """
    start_idx = max(0, env.current_step - window)
    realized_returns = np.dot(env.returns.iloc[start_idx:env.current_step].values, env.previous_weights)
    sortino = RiskMetricsComputer.compute_rolling_sortino(realized_returns, risk_free_rate, window)
    return sortino


def compute_max_drawdown_reward(env, lookback: int = 50, scaling_factor: float = 1.0) -> float:
    """
    Compute a reward based on the inverse of the maximum drawdown.

    Args:
        env (PortfolioEnv): The environment instance.
        lookback (int): Number of steps to look back for drawdown computation.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Inverse of maximum drawdown (higher is better).
    """
    if len(env.portfolio_history) < lookback:
        recent_values = np.array(env.portfolio_history)
    else:
        recent_values = np.array(env.portfolio_history[-lookback:])
    max_dd = RiskMetricsComputer.max_drawdown(recent_values)
    # To convert drawdown to a reward where higher is better
    inverse_drawdown = 1.0 / (max_dd + 1e-8)
    return inverse_drawdown


def compute_calmar_ratio_reward(env, window: int = 252, risk_free_rate: float = 0.02, scaling_factor: float = 1.0) -> float:
    """
    Compute the Calmar ratio as a reward.

    Args:
        env (PortfolioEnv): The environment instance.
        window (int): Number of steps to consider for computation.
        risk_free_rate (float): Annual risk-free rate.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Calmar ratio.
    """
    if len(env.portfolio_history) < window:
        portfolio_values = np.array(env.portfolio_history)
    else:
        portfolio_values = np.array(env.portfolio_history[-window:])
    calmar = RiskMetricsComputer.calmar_ratio(portfolio_values, risk_free_rate)
    return calmar


def compute_var(env, window: int = 20, confidence: float = 0.95, scaling_factor: float = 1.0) -> float:
    """
    Compute the Value at Risk (VaR) as a negative reward.

    Args:
        env (PortfolioEnv): The environment instance.
        window (int): Number of steps to consider for VaR computation.
        confidence (float): Confidence level.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Negative VaR (lower VaR means higher penalty).
    """
    start_idx = max(0, env.current_step - window)
    realized_returns = np.dot(env.returns.iloc[start_idx:env.current_step].values, env.previous_weights)
    var = RiskMetricsComputer.compute_rolling_var(realized_returns, window, confidence)
    return -var  # Penalize high VaR


def compute_cvar(env, window: int = 20, confidence: float = 0.95, scaling_factor: float = 1.0) -> float:
    """
    Compute the Conditional Value at Risk (CVaR) as a negative reward.

    Args:
        env (PortfolioEnv): The environment instance.
        window (int): Number of steps to consider for CVaR computation.
        confidence (float): Confidence level.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Negative CVaR (higher CVaR means higher penalty).
    """
    start_idx = max(0, env.current_step - window)
    realized_returns = np.dot(env.returns.iloc[start_idx:env.current_step].values, env.previous_weights)
    cvar = RiskMetricsComputer.compute_rolling_cvar(realized_returns, window, confidence)
    return -cvar  # Penalize high CVaR


def compute_average_return(env, window: int = 20, scaling_factor: float = 1.0) -> float:
    """
    Compute the average return over the last `window` steps as a reward.

    Args:
        env (PortfolioEnv): The environment instance.
        window (int): Number of steps to consider.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Average return.
    """
    start_idx = max(0, env.current_step - window)
    realized_returns = np.dot(env.returns.iloc[start_idx:env.current_step].values, env.previous_weights)
    avg_return = np.mean(realized_returns)
    return avg_return


def compute_return_variance(env, window: int = 20, scaling_factor: float = 1.0) -> float:
    """
    Compute the negative variance of returns over the last `window` steps as a penalty.

    Args:
        env (PortfolioEnv): The environment instance.
        window (int): Number of steps to consider.
        scaling_factor (float): Scaling factor for normalization.

    Returns:
        float: Negative variance (higher variance means higher penalty).
    """
    start_idx = max(0, env.current_step - window)
    realized_returns = np.dot(env.returns.iloc[start_idx:env.current_step].values, env.previous_weights)
    variance = np.var(realized_returns)
    return -variance  # Penalize high variance
