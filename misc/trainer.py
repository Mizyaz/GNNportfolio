# trainer.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from financial_env import PortfolioEnv
from metrics_computer import MetricsComputer, MetricsConfig
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import warnings
import pickle 
import time
import os

class TrainingCallback(gym.Wrapper):
    """
    Custom callback for logging metrics during training.
    """
    def __init__(self, env, verbose=0):
        super(TrainingCallback, self).__init__(env)
        self.episode_rewards = []
        self.episode_steps = []
    
    def reset(self, **kwargs):
        self.episode_rewards = []
        self.episode_steps = []
        self.counter = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self.episode_rewards.append(reward)
        self.episode_steps.append(1)
        if done and self.counter % 50 == 0:
            cumulative_return = self.env.portfolio_value / self.env.initial_balance - 1
            print(f"Episode ended. Cumulative Return: {cumulative_return:.2%}")
            self.episode_rewards = []
            self.episode_steps = []
        self.counter += 1
        return observation, reward, done, truncated, info

class PortfolioTrainer:
    """
    Trainer class to train an RL agent for portfolio optimization.
    """
    def __init__(self, env_config: Dict[str, Any], ppo_config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment.
            ppo_config (Dict[str, Any]): Configuration dictionary for the PPO model.
        """
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.model = None
        self.env = None
        self.name = ppo_config.get('name', 'nameless')
    
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
    
    def compute_equal_weight_cumulative_return(self, returns: pd.DataFrame) -> float:
        """
        Compute the cumulative return of an equal-weighted portfolio.
        
        Args:
            returns (pd.DataFrame): Daily returns of assets.
        
        Returns:
            float: Cumulative return.
        """
        print("Computing equal-weighted cumulative return...")
        equal_weights = np.ones(self.env_config['num_assets']) / self.env_config['num_assets']
        portfolio_returns = returns.dot(equal_weights)
        cumulative_return = (1 + portfolio_returns).prod() - 1
        print(f"Equal-Weight Cumulative Return: {cumulative_return:.2%}")
        return cumulative_return
    
    def initialize_environment(self, prices: pd.DataFrame, returns: pd.DataFrame, test_prices: pd.DataFrame, test_returns: pd.DataFrame):
        """
        Initialize the custom environment.
        
        Args:
            prices (pd.DataFrame): Adjusted close prices.
            returns (pd.DataFrame): Daily returns.
        """
        # Initialize the environment
        self.env = PortfolioEnv(config=self.env_config, prices=prices, returns=returns)

        self.test_env = PortfolioEnv(config=self.env_config, prices=test_prices, returns=test_returns)
        
        # Wrap the environment with a callback for logging
        self.env = TrainingCallback(self.env)
        
        # Check the environment
        print("Checking environment compatibility...")
        check_env(self.env, warn=True)
        print("Environment is compatible.")
    
    def train(self, prices: pd.DataFrame, returns: pd.DataFrame, test_prices: pd.DataFrame, test_returns: pd.DataFrame, total_timesteps: int = 100000):
        """
        Train the PPO model.
        
        Args:
            prices (pd.DataFrame): Adjusted close prices.
            returns (pd.DataFrame): Daily returns.
            total_timesteps (int): Number of timesteps to train.
        """
        # Initialize the environment
        self.initialize_environment(prices, returns, test_prices, test_returns)
        
        # Initialize the PPO model
        print("Initializing PPO model...")
        self.model = PPO(
            policy=self.ppo_config.get('policy', 'MlpPolicy'),
            env=self.env,
            verbose=self.ppo_config.get('verbose', 1),
            learning_rate=self.ppo_config.get('learning_rate', 1e-4),
            n_steps=self.ppo_config.get('n_steps', 2048),
            batch_size=self.ppo_config.get('batch_size', 64),
            n_epochs=self.ppo_config.get('n_epochs', 10),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_range=self.ppo_config.get('clip_range', 0.2),
            ent_coef=self.ppo_config.get('ent_coef', 0.0),
        )
        print("PPO model initialized.")
        
        # Train the model
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")

        # single episode
        obs, info = self.env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        while not done and not truncated:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
        total_return = self.env.portfolio_value / self.env.initial_balance - 1
        print(f"Single episode total return: {total_return:.2%}")
        
        # Save the trained model
        self.model.save(f"{self.name}_ppo_portfolio_optimizer")

        print(f"Model saved as '{self.name}_ppo_portfolio_optimizer.zip'")
    
        with open(f"{self.name}_ppo_config.pkl", "wb") as f:
            pickle.dump(self.ppo_config, f)

        with open(f"{self.name}_env_config.pkl", "wb") as f:
            pickle.dump(self.env_config, f)

        return total_return

    def evaluate(self, returns: pd.DataFrame, episodes: int = 5):
        """
        Evaluate the trained model.
        
        Args:
            returns (pd.DataFrame): Daily returns of assets.
            episodes (int): Number of episodes to evaluate.
        """
        print(f"Evaluating the model over {episodes} episodes...")

        total_return_list = []

        print(f"test env equal weight return: {self.compute_equal_weight_cumulative_return(returns):.2%}")
        for episode in range(episodes):
            obs, info = self.test_env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            while not done and not truncated:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.test_env.step(action)
                total_reward += reward
            total_return = self.test_env.portfolio_value / self.test_env.initial_balance - 1
            total_return_list.append(total_return)
            print(f"Episode {episode + 1}: Total Return: {total_return:.2%}")
        
        print(f"Average Total Return: {np.mean(total_return_list):.2%}")
        
        # Render the last episode's performance
        print("Rendering portfolio performance...")
        #self.test_env.render()
        return total_return_list
    
    def run(self, tickers: List[str], start_date: str, end_date: str, test_start_date: str, test_end_date: str, total_timesteps: int = 100000, episodes: int = 5):
        """
        Full run: download data, compute equal-weighted return, train, evaluate.
        
        Args:
            tickers (List[str]): List of asset ticker symbols.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            total_timesteps (int): Number of timesteps to train.
            episodes (int): Number of episodes to evaluate.
        """
        # Download data
        prices, returns = self.download_data(tickers, start_date, end_date)

        test_prices, test_returns = self.download_data(tickers, test_start_date, test_end_date)
        
        # Compute and print equal-weighted cumulative return
        print("equal weight return of training data: ", self.compute_equal_weight_cumulative_return(returns))
        print("equal weight return of test data: ", self.compute_equal_weight_cumulative_return(test_returns))
        
        # Train the model
        total_return = self.train(prices, returns, test_prices, test_returns, total_timesteps)
        
        # Evaluate the model
        total_return_list = self.evaluate(returns, episodes)

        return total_return, max(total_return_list)

def train_and_evaluate():
    # Define asset tickers (example with some major tech stocks)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'NVDA', 'TSLA', 'JPM', 'JNJ', 'PG']
    
    # Time period for training
    start_date = "2017-01-01"
    end_date = "2018-01-01"

    test_start_date = "2018-01-01"
    test_end_date = "2022-01-01"
    
    # Environment configuration
    env_config = {
        'window_size': 30,  # Number of days to look back
        'num_assets': len(tickers),
        'risk_free_rate': 0.02,  # 2% annual risk-free rate
        'metrics_list': [
            ('sharpe_ratio', 0.03),          # 30% weight
            ('sortino_ratio', 0.02),         # 20% weight
            ('maximum_drawdown', -0.02),      # 20% weight (negative weight as we want to minimize)
            ('portfolio_volatility', -0.01),  # 10% weight (negative weight as we want to minimize)
            ('cumulative_returns', 0.02),     # 20% weight
            ('macd_hist', 0.01),
            ('rsi', 0.01),
        ],
        'compute_hht': True,
        'extensive_metrics': True,
    }
    
    # PPO model configuration
    ppo_config = {
        'policy': 'MultiInputPolicy',
        'name': 'hilbert',
        'verbose': 0,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,  # Slightly increase exploration
    }
    
    # Training parameters
    training_params = {
        'total_timesteps': 50000,  # Total timesteps for training
        'eval_episodes': 5,         # Number of episodes for evaluation
    }

    for extensive_metrics in [True, False]:
        for compute_hht in [False, True]:
            env_config['compute_hht'] = compute_hht
            env_config['extensive_metrics'] = extensive_metrics

            print("test with hht: ", env_config['compute_hht'], " and extensive metrics: ", env_config['extensive_metrics'])

            start_time = time.time()
            
            # Initialize and run the trainer
            trainer = PortfolioTrainer(env_config=env_config, ppo_config=ppo_config)
            train_return, test_return = trainer.run(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                total_timesteps=training_params['total_timesteps'],
                episodes=training_params['eval_episodes']
            )
            end_time = time.time()
            print(f"Total time taken: {end_time - start_time:.2f} seconds")

            print(f"Train return: {train_return:.2%}")
            print(f"Test return: {test_return:.2%}")

if __name__ == "__main__":
    train_and_evaluate()
