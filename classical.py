import numpy as np
import pandas as pd
from typing import List, Tuple
from pypfopt import EfficientFrontier, objective_functions, risk_models, expected_returns, BlackLittermanModel
from pypfopt.risk_models import CovarianceShrinkage

from trainer import download_data

# Utility function to calculate portfolio performance
def calculate_portfolio_performance(weights, data):
    """
    Calculate portfolio value history and final percentage return.
    """
    prices = data['Close']
    portfolio_daily_returns = (prices.pct_change().fillna(0) @ weights)
    portfolio_value = (1 + portfolio_daily_returns).cumprod()
    final_return = portfolio_value.iloc[-1] - 1  # Final percentage return
    return final_return, portfolio_value.values.tolist()

# Markowitz Optimization
def optimize_markowitz(data: pd.DataFrame):
    prices = data['Close']
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()  # Maximize Sharpe Ratio
    cleaned_weights = ef.clean_weights()
    return calculate_portfolio_performance(np.array(list(cleaned_weights.values())), data)

# Black-Litterman
def optimize_black_litterman(data: pd.DataFrame, market_caps: pd.Series = None, views: dict = None, confidences: dict = None):
    prices = data['Close']
    mu = expected_returns.mean_historical_return(prices)

    views = {
        'AAPL': 0.1,
        'MSFT': 0.1,
        'GOOG': 0.1,
        'AMZN': 0.1
    }
    confidences = {
        'AAPL': 0.1,
        'MSFT': 0.1,
        'GOOG': 0.1,
        'AMZN': 0.1
    }
    S = risk_models.sample_cov(prices)
    Q = np.array(list(views.values()))
    P = np.array(list(views.keys()))
    bl = BlackLittermanModel(S, pi=mu, market_caps=market_caps, absolute_views=views, omega=confidences, risk_aversion=1, Q=Q, P=P)
    bl_return = bl.bl_returns()
    ef = EfficientFrontier(bl_return, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return calculate_portfolio_performance(np.array(list(cleaned_weights.values())), data)

# Minimum Variance
def optimize_min_variance(data: pd.DataFrame):
    prices = data['Close']
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(None, S)
    weights = ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    return calculate_portfolio_performance(np.array(list(cleaned_weights.values())), data)

# Maximum Sharpe Ratio
def optimize_max_sharpe(data: pd.DataFrame):
    prices = data['Close']
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return calculate_portfolio_performance(np.array(list(cleaned_weights.values())), data)

# Example Usage
if __name__ == "__main__":
    from pathlib import Path
    import pickle
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    data, prices, returns, high_prices, low_prices, volumes = download_data(tickers, start_date, end_date, return_df=True)
    # Example calls
    markowitz_result = optimize_markowitz(data)
    min_variance_result = optimize_min_variance(data)
    max_sharpe_result = optimize_max_sharpe(data)
    
    print("Markowitz Optimization:", markowitz_result[0])
    print("Minimum Variance Optimization:", min_variance_result[0])
    print("Maximum Sharpe Ratio Optimization:", max_sharpe_result[0])
