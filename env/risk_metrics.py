# risk_metrics.py

import numpy as np

class RiskMetricsComputer:
    """Computes various risk metrics for the portfolio."""
    
    @staticmethod
    def compute_rolling_volatility(returns: np.ndarray, window: int = 20) -> float:
        if len(returns) < window:
            return 0.0
        recent = returns[-window:]
        return np.std(recent) * np.sqrt(252)
    
    @staticmethod
    def compute_rolling_sharpe(returns: np.ndarray, risk_free_rate: float, window: int = 20) -> float:
        if len(returns) < window:
            return 0.0
        excess = returns[-window:] - risk_free_rate/252
        return (np.mean(excess) / (np.std(excess) + 1e-8)) * np.sqrt(252)
    
    @staticmethod
    def compute_rolling_sortino(returns: np.ndarray, risk_free_rate: float, window: int = 20) -> float:
        if len(returns) < window:
            return 0.0
        excess = returns[-window:] - risk_free_rate/252
        downside = np.where(excess < 0, excess, 0)
        downside_std = np.std(downside) + 1e-8
        return (np.mean(excess) / downside_std) * np.sqrt(252)
    
    @staticmethod
    def compute_rolling_var(returns: np.ndarray, window: int = 20, confidence: float = 0.95) -> float:
        if len(returns) < window:
            return 0.0
        recent = returns[-window:]
        return -np.percentile(recent, (1 - confidence) * 100)
    
    @staticmethod
    def compute_rolling_cvar(returns: np.ndarray, window: int = 20, confidence: float = 0.95) -> float:
        if len(returns) < window:
            return 0.0
        recent = returns[-window:]
        var = -np.percentile(recent, (1 - confidence) * 100)
        cvar_values = recent[recent <= -var]
        if len(cvar_values) == 0:
            return 0.0
        return -np.mean(cvar_values)
    
    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """Compute maximum drawdown from a series of portfolio values."""
        if len(portfolio_values) < 2:
            return 0.0
        peak = portfolio_values[0]
        max_dd = 0.0
        for val in portfolio_values:
            if val > peak:
                peak = val
            drawdown = (peak - val) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd
    
    @staticmethod
    def calmar_ratio(portfolio_values: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Compute Calmar ratio: annualized return / max drawdown."""
        if len(portfolio_values) < 2:
            return 0.0
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        # Approx annual factor if daily data
        years = len(portfolio_values) / 252.0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        mdd = RiskMetricsComputer.max_drawdown(portfolio_values)
        return annual_return / (mdd + 1e-8)
