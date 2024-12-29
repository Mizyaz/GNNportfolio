import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std_dev, returns

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_std_dev, p_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_std_dev
    return -sharpe_ratio

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = minimize(neg_sharpe, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def risk_parity(cov_matrix):
    """Calculates risk parity portfolio weights."""
    num_assets = len(cov_matrix)
    # Calculate the inverse of the volatilities
    inv_volatilities = 1.0 / np.sqrt(np.diag(cov_matrix))
    # Normalize to get weights
    weights = inv_volatilities / np.sum(inv_volatilities)
    return weights

def optimal_portfolio(tickers, start_date, end_date, risk_free_rate=0.0, tickers_from_file=False, initial_investment=1000):
    if tickers_from_file:
        try:
            tickers = pd.read_csv(tickers, header=None).values.flatten().tolist()
        except FileNotFoundError:
            print(f"Error: Ticker file '{tickers}' not found.")
            return
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if data.empty:
        print("No data downloaded. Check your tickers and date range.")
        return

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    # ... (Max Sharpe and Min Variance calculations as before)
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    min_var = min_variance(mean_returns, cov_matrix)

    # Risk Parity Portfolio
    risk_parity_weights = risk_parity(cov_matrix)

    # Equal Weight Portfolio
    equal_weights = np.array([1/num_assets] * num_assets)

    # Portfolio Value Over Time (for all portfolios)
    portfolio_values = pd.DataFrame(index=returns.index)
    portfolio_values['Max Sharpe'] = (1 + returns @ max_sharpe['x']).cumprod() * initial_investment
    portfolio_values['Min Variance'] = (1 + returns @ min_var['x']).cumprod() * initial_investment
    portfolio_values['Risk Parity'] = (1 + returns @ risk_parity_weights).cumprod() * initial_investment
    portfolio_values['Equal Weight'] = (1 + returns @ equal_weights).cumprod() * initial_investment

    #Output

    sdp_ew, rp_ew = portfolio_annualised_performance(equal_weights, mean_returns, cov_matrix)
    equal_weight_allocation = pd.DataFrame(equal_weights, index=data.columns, columns=['allocation'])
    equal_weight_allocation['allocation'] = [round(i*100,2) for i in equal_weight_allocation['allocation']]

    print("-"*80)
    print("Equal Weight Portfolio Allocation\n")
    print("Annualised Return:", round(rp_ew,2))
    print("Annualised Volatility:", round(sdp_ew,2))
    print("\n")
    print(equal_weight_allocation)
    print("-"*80)



    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend(portfolio_values.columns)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage (same as before)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']  # Default tickers
    start_date = '2021-10-26'
    end_date = '2023-10-26'
    tickers_file = 'sp500tickers.txt' # Path to your tickers file

    # Run with default tickers
    optimal_portfolio(tickers, start_date, end_date)
