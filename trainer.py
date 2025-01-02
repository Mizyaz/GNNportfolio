# train.py

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.portfolio_env import *
from env.config import *
import pickle
from typing import Tuple, Optional
from pathlib import Path
import yfinance as yf
from classical import *
import time
from env.reward import *
from sb3_contrib import RecurrentPPO, TRPO

def download_data(tickers: List[str], start_date: str, end_date: str, return_df: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and preprocess data from yfinance with caching"""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create cache filename based on parameters
    cache_file = data_dir / f"market_data_{'-'.join(sorted(tickers))}_{start_date}_{end_date}.pkl"
    
    # Check if cached data exists
    if cache_file.exists():
        print("Loading data from cache...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            
        # Verify data structure
        expected_columns = ['Close', 'High', 'Low', 'Volume']
        if all(col in cached_data.columns.levels[0] for col in expected_columns):
            print("Cache validation successful.")
            data = cached_data
        else:
            print("Cache validation failed. Downloading fresh data...")
            data = yf.download(tickers, start=start_date, end=end_date)
    else:
        print(f"Downloading data for {len(tickers)} assets...")
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Cache the raw data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print("Data cached for future use.")
    
    # Extract and process data
    if return_df:
        prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        volumes = data['Volume']
    else:
        prices = data['Close'].astype(np.float64)
        high_prices = data['High'].astype(np.float64)
        low_prices = data['Low'].astype(np.float64)
        volumes = data['Volume'].astype(np.float64)
        
    # Calculate returns using close prices
    returns = prices.pct_change()
    
    # Handle missing data
    for df in [prices, returns, high_prices, low_prices, volumes]:
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
    
    print("Data preprocessing complete.")

    if return_df:
        return data, prices, returns, high_prices, low_prices, volumes
    else:
        return prices, returns, high_prices, low_prices, volumes


def main():
    # Paths to your data files

    initial_balance = 100

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "DIS", "WMT"]
    data, prices, returns, high_prices, low_prices, volumes = download_data(tickers, "2016-01-01", "2017-01-01", return_df=True)

    existing_results = {
        "markowitz": (optimize_markowitz(data)[0], np.array(optimize_markowitz(data)[1])*initial_balance),
        "min_variance": (optimize_min_variance(data)[0], np.array(optimize_min_variance(data)[1])*initial_balance),
        "max_sharpe": (optimize_max_sharpe(data)[0], np.array(optimize_max_sharpe(data)[1])*initial_balance)
    }

    prices, returns, high_prices, low_prices, volumes = download_data(tickers, "2016-01-01", "2017-01-01", return_df=False)

    # Ensure that all dataframes have the same index
    if not all(df.index.equals(prices.index) for df in [returns, high_prices, low_prices, volumes] if df is not None):
        raise ValueError("All input dataframes must have the same index (dates).")

    # Define portfolio configuration
    config = PortfolioConfig(
        window_size=7,
        num_assets=prices.shape[1],
        initial_balance=initial_balance,
        risk_free_rate=0.02,
        transaction_cost=0.001,
        use_technical_indicators=True,
        use_correlation_features=True,
        use_risk_metrics=True,
        use_time_freq=False,
        reward_config=RewardConfig(
            rewards=[
                (compute_sharpe, 0.4, {"window": 7, "scaling_factor": 0.5}),
                (compute_sortino, 0.1, {"window": 7, "scaling_factor": 0.5}),
                (compute_max_drawdown_reward, 0.1, {"lookback": 7, "scaling_factor": 0.2}),
                (compute_average_return, 0.4, {"window": 7, "scaling_factor": 0.5}),
            ]
        ),
        use_per_episode_plot=False
    )

    # Initialize the environment
    env = PortfolioEnv(
        config=config,
        prices=prices,
        returns=returns,
        high_prices=high_prices,
        low_prices=low_prices,
        volumes=volumes,
        existing_results=existing_results
    )

    # Wrap the environment
    vec_env = DummyVecEnv([lambda: env])

    start_time = time.time()
    # Initialize the RL agent (PPO with MultiInputPolicy to handle Dict observation space)
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_portfolio_tensorboard/",
        learning_rate=0.001
    )

    # Define training parameters
    total_timesteps = 1_000_000  # Adjust based on your computational resources
    log_interval = 10          # Log every 10 episodes

    # Training loop with periodic evaluation
    for i in range(total_timesteps // 10000):
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        print(f"Completed {(i+1)*10000} timesteps")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Save the trained model
    model_path = "ppo_portfolio_model"
    model.save(model_path)
    print(f"Trained model saved to {model_path}")


    # Evaluate the trained agent
    obs, _ = env.reset()
    done = False
    portfolio_history_rl = []
    start_time = time.time()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_history_rl.append(env.portfolio_value)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.2f} seconds")

    total_return_rl = (env.portfolio_value / config.initial_balance) - 1
    print(f"Trained Agent Total Return: {total_return_rl * 100:.2f}%")

    # Run the equal-weight baseline
    total_return_eq, portfolio_history_eq = run_equal_weight_baseline(prices, returns, config)
    print(f"Equal-Weight Baseline Total Return: {total_return_eq * 100:.2f}%")

    # Save the results
    results = {
        "trained_agent": {
            "total_return": total_return_rl,
            "portfolio_history": portfolio_history_rl
        },
        "equal_weight_baseline": {
            "total_return": total_return_eq,
            "portfolio_history": portfolio_history_eq
        },
        "training_time": training_time,
        "inference_time": inference_time
    }

    with open("training_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Training results saved to training_results.pkl")

    # Optionally, save the portfolio histories as CSV for further analysis
    pd.DataFrame({
        "Trained_Agent": np.pad(portfolio_history_rl, (0, 100000 - len(portfolio_history_rl)), mode='constant', constant_values=0),
        "Equal_Weight": np.pad(portfolio_history_eq, (0, 100000 - len(portfolio_history_eq)), mode='constant', constant_values=0)
    }).to_csv("portfolio_histories.csv", index=False)
    print("Portfolio histories saved to portfolio_histories.csv")

if __name__ == "__main__":
    main()
