
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf
import matplotlib.pyplot as plt
from experimental import max_sharpe_ratio, min_variance, risk_parity
import os
import seaborn as sns

from tfr_financial_env import SpectralPortfolioEnv

def train_spectral_agent(data, start_train_idx, end_train_idx, start_test_idx, end_test_idx, risk_free_rate=0.0):
    """Trains a PPO agent using spectral observations."""
    env_train = lambda: SpectralPortfolioEnv(
        data.iloc[start_train_idx:end_train_idx],
        0,
        len(data.iloc[start_train_idx:end_train_idx]) - 1,
        risk_free_rate
    )
    env_train = DummyVecEnv([env_train])

    # Modified network architecture for spectral inputs
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 64], vf=[128, 64])]
    )

    model = PPO(
        'MlpPolicy',
        env_train,
        verbose=1,
        learning_rate=0.0001,
        gamma=0.99,
        policy_kwargs=policy_kwargs
    )
    
    model.learn(total_timesteps=500000)
    return model

def main():
    """
    Main function implementing spectral portfolio optimization with comprehensive evaluation.
    Compares traditional strategies against the spectral RL approach.
    """
    # Configuration parameters
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    start_date = '2021-01-01'
    end_date = '2023-12-31'
    risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
    initial_investment = 10000
    
    print("Initializing Portfolio Optimization Analysis...")
    print(f"Assets: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    
    # Data acquisition and preprocessing
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if data.empty:
            raise ValueError("No data downloaded")
            
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(returns.columns)
        
        print(f"\nData successfully loaded: {len(data)} trading days")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Train-test split
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    print("\nComputing traditional portfolio optimization strategies...")
    
    # Initialize portfolio tracker
    portfolio_values = pd.DataFrame(index=test_data.index)
    
    # 1. Equal Weight Strategy
    equal_weights = np.array([1/num_assets] * num_assets)
    portfolio_values['Equal Weight'] = (1 + test_data.pct_change().dropna() @ equal_weights).cumprod() * initial_investment
    
    # 2. Maximum Sharpe Ratio Strategy
    try:
        max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
        portfolio_values['Max Sharpe'] = (1 + test_data.pct_change().dropna() @ max_sharpe['x']).cumprod() * initial_investment
        print("Maximum Sharpe Ratio weights computed")
    except Exception as e:
        print(f"Error computing Max Sharpe: {str(e)}")
    
    # 3. Minimum Variance Strategy
    try:
        min_var = min_variance(mean_returns, cov_matrix)
        portfolio_values['Min Variance'] = (1 + test_data.pct_change().dropna() @ min_var['x']).cumprod() * initial_investment
        print("Minimum Variance weights computed")
    except Exception as e:
        print(f"Error computing Min Variance: {str(e)}")
    
    # 4. Risk Parity Strategy
    try:
        risk_parity_weights = risk_parity(cov_matrix)
        portfolio_values['Risk Parity'] = (1 + test_data.pct_change().dropna() @ risk_parity_weights).cumprod() * initial_investment
        print("Risk Parity weights computed")
    except Exception as e:
        print(f"Error computing Risk Parity: {str(e)}")
    
    print("\nTraining Spectral RL Agent...")
    
    # Train spectral RL agent
    try:
        model = train_spectral_agent(data, 0, split_index-1, split_index, len(data)-1, risk_free_rate)
        
        # Evaluate RL agent on test set
        env_test = SpectralPortfolioEnv(test_data, 0, len(test_data)-1, risk_free_rate)
        obs, _ = env_test.reset()
        done = False
        rl_portfolio_values = [initial_investment]
        rl_weights_history = []
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env_test.step(action)
            rl_portfolio_values.append(info['portfolio_value'])
            rl_weights_history.append(info['weights'])
            
        portfolio_values['Spectral RL'] = pd.Series(rl_portfolio_values[1:], index=portfolio_values.index)
        print("RL agent evaluation completed")
        
    except Exception as e:
        print(f"Error in RL training/evaluation: {str(e)}")
    
    # Performance Analysis
    print("\nComputing performance metrics...")
    
    performance_metrics = pd.DataFrame(columns=['Total Return (%)', 'Annual Return (%)',
                                              'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)'])
    
    for strategy in portfolio_values.columns:
        returns_series = portfolio_values[strategy].pct_change().dropna()
        total_return = (portfolio_values[strategy].iloc[-1] / initial_investment - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(returns_series)) - 1
        volatility = returns_series.std() * np.sqrt(252) * 100
        sharpe = (annual_return - risk_free_rate) / (volatility/100) if volatility != 0 else 0
        drawdown = ((portfolio_values[strategy].cummax() - portfolio_values[strategy]) / 
                   portfolio_values[strategy].cummax()).max() * 100
        
        performance_metrics.loc[strategy] = [
            round(total_return, 2),
            round(annual_return * 100, 2),
            round(volatility, 2),
            round(sharpe, 2),
            round(drawdown, 2)
        ]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Portfolio Values Plot
    portfolio_values.plot(ax=ax1)
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)
    
    # Performance Metrics Plot
    performance_metrics[['Total Return (%)', 'Sharpe Ratio']].plot(kind='bar', ax=ax2)
    ax2.set_title('Strategy Performance Comparison')
    ax2.set_ylabel('Metric Value')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed performance metrics
    print("\nDetailed Performance Metrics:")
    print(performance_metrics.to_string())
    
    # Save results
    try:
        results_dir = "portfolio_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save performance metrics
        performance_metrics.to_csv(f"{results_dir}/performance_metrics.csv")
        
        # Save portfolio values
        portfolio_values.to_csv(f"{results_dir}/portfolio_values.csv")
        
        print(f"\nResults saved to {results_dir}/")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main()