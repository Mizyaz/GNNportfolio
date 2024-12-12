# trainer.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from financial_env import PortfolioEnv
from tfr_financial_env import TFRPortfolioEnv
from metrics_computer import MetricsComputer, MetricsConfig
from time_frequency_analyser import TimeFrequencyAnalyser, TFAConfig
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import warnings

class PortfolioTrainer:
    """
    Trainer class to train RL agents for portfolio optimization.
    """
    def __init__(self, env_type: str = 'basic'):
        """
        Initialize the trainer.
        
        Args:
            env_type (str): Type of environment ('basic' or 'tfr').
        """
        self.env_type = env_type
        self.model = None
        self.env = None
    
    def download_data(self, tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download and preprocess data from yfinance.
        
        Args:
            tickers (List[str]): List of asset ticker symbols.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Adjusted close prices and daily returns.
        """
        print("Downloading data from Yahoo Finance...")
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data.dropna(inplace=True)
        returns = data.pct_change().dropna()
        print("Data download complete.")
        return data, returns
    
    def compute_equal_weight_cumulative_return(self, returns: pd.DataFrame, num_assets: int) -> float:
        """
        Compute the cumulative return of an equal-weighted portfolio.
        
        Args:
            returns (pd.DataFrame): Daily returns of assets.
            num_assets (int): Number of assets in the portfolio.
        
        Returns:
            float: Cumulative return.
        """
        print("Computing equal-weighted cumulative return...")
        equal_weights = np.ones(num_assets) / num_assets
        portfolio_returns = returns.dot(equal_weights)
        cumulative_return = (1 + portfolio_returns).prod() - 1
        print(f"Equal-Weight Cumulative Return: {cumulative_return:.2%}")
        return cumulative_return
    
    def initialize_environment(self, config: Dict[str, Any], prices: pd.DataFrame, returns: pd.DataFrame):
        """
        Initialize the chosen environment.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary for the environment.
            prices (pd.DataFrame): Adjusted close prices.
            returns (pd.DataFrame): Daily returns.
        """
        if self.env_type == 'basic':
            self.env = PortfolioEnv(config=config, prices=prices, returns=returns)
        elif self.env_type == 'tfr':
            self.env = TFRPortfolioEnv(config=config, prices=prices, returns=returns)
        else:
            raise ValueError("Unsupported environment type. Choose 'basic' or 'tfr'.")
        
        # Check the environment
        print("Checking environment compatibility...")
        check_env(self.env, warn=True)
        print("Environment is compatible.")
    
    def train(self, config: Dict[str, Any], prices: pd.DataFrame, returns: pd.DataFrame, ppo_config: Dict[str, Any], total_timesteps: int = 100000):
        """
        Train the PPO model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary for the environment.
            prices (pd.DataFrame): Adjusted close prices.
            returns (pd.DataFrame): Daily returns.
            ppo_config (Dict[str, Any]): Configuration dictionary for the PPO model.
            total_timesteps (int): Number of timesteps to train.
        """
        # Initialize the environment
        self.initialize_environment(config, prices, returns)
        
        # Initialize the PPO model
        print("Initializing PPO model...")
        self.model = PPO(
            policy=ppo_config.get('policy', 'MlpPolicy'),
            env=self.env,
            verbose=ppo_config.get('verbose', 1),
            learning_rate=ppo_config.get('learning_rate', 1e-4),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.0),
        )
        print("PPO model initialized.")
        
        # Train the model
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")
        
        # Save the trained model
        self.model.save("ppo_portfolio_optimizer")
        print("Model saved as 'ppo_portfolio_optimizer.zip'")
    
    def evaluate(self, config: Dict[str, Any], prices: pd.DataFrame, returns: pd.DataFrame, episodes: int = 5):
        """
        Evaluate the trained model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary for the environment.
            prices (pd.DataFrame): Adjusted close prices.
            returns (pd.DataFrame): Daily returns.
            episodes (int): Number of episodes to evaluate.
        """
        # Re-initialize the environment for evaluation
        self.initialize_environment(config, prices, returns)
        
        print(f"Evaluating the model over {episodes} episodes...")
        for episode in range(episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            while not done and not truncated:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Return: {total_reward:.2%}")
        
        # Render the last episode's performance
        print("Rendering portfolio performance...")
        self.env.render()
    
    def run(self, tickers: List[str], start_date: str, end_date: str, config: Dict[str, Any], ppo_config: Dict[str, Any], total_timesteps: int = 100000, episodes: int = 5):
        """
        Full run: download data, compute equal-weighted return, train, evaluate.
        
        Args:
            tickers (List[str]): List of asset ticker symbols.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            config (Dict[str, Any]): Configuration dictionary for the environment.
            ppo_config (Dict[str, Any]): Configuration dictionary for the PPO model.
            total_timesteps (int): Number of timesteps to train.
            episodes (int): Number of episodes to evaluate.
        """
        # Download data
        prices, returns = self.download_data(tickers, start_date, end_date)
        
        # Compute and print equal-weighted cumulative return
        self.compute_equal_weight_cumulative_return(returns, config.get('num_assets', len(tickers)))
        
        # Train the model
        self.train(config, prices, returns, ppo_config, total_timesteps)
        
        # Evaluate the model
        self.evaluate(config, prices, returns, episodes)

def main():
    """
    Example usage of the PortfolioTrainer with Time-Frequency Environment.
    """
    # Define asset tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'PG', 'MA']
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    
    # Define environment configuration
    env_config = {
        'window_size': 20,
        'num_assets': 10,
        'metrics_list': [
            ('Sharpe_ratio', 1.0),
            ('Sortino_ratio', 0.5),
            ('Maximum_drawdown', -0.5),
            ('Cumulative_returns', 1.0),
            ('Portfolio_volatility', -0.3),
            ('Alpha', 0.2),
            ('Beta', 0.1),
            ('Turnover_rate', -0.2),
            ('Information_ratio', 0.3),
            ('Diversification_metrics', 0.4),
            ('Value_at_risk', -0.4),
            ('Conditional_value_at_risk', -0.4),
            ('Transaction_costs', -0.3),
            ('Liquidity_metrics', 0.2),
            ('Exposure_metrics', 0.2),
            ('Drawdown_duration', -0.3)
        ],
        'risk_free_rate': 0.02,
        'tfa_config': {
            'n_mels': 40,
            'n_fft': 2048,
            'hop_length': 512,
            'window': 'hann',
            'fmax': None,
            'entropy_bins': 10
        }
    }
    
    # Define PPO configuration
    ppo_config = {
        'policy': 'MultiInputPolicy',
        'verbose': 1,
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0
    }
    
    # Initialize the trainer with Time-Frequency Environment
    trainer = PortfolioTrainer(env_type='tfr')
    
    # Run the training and evaluation
    trainer.run(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        config=env_config,
        ppo_config=ppo_config,
        total_timesteps=100000,
        episodes=5
    )

if __name__ == "__main__":
    main()
