import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe Ratio."""
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    return sharpe

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate the Maximum Drawdown."""
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_performance_metrics(portfolio_returns: pd.DataFrame):
    """Calculate and print performance metrics."""
    strategy_daily = portfolio_returns['GAT Strategy'].pct_change().dropna()
    equal_daily = portfolio_returns['Equal-Weighted Portfolio'].pct_change().dropna()

    metrics = {
        'Strategy Sharpe': calculate_sharpe_ratio(strategy_daily),
        'Equal-Weight Sharpe': calculate_sharpe_ratio(equal_daily),
        'Strategy Max Drawdown': calculate_max_drawdown(portfolio_returns['GAT Strategy']),
        'Equal-Weight Max Drawdown': calculate_max_drawdown(portfolio_returns['Equal-Weighted Portfolio'])
    }
    
    return metrics 