#!/usr/bin/env python3
"""
train_rl.py

This script defines a reinforcement learning environment using the FinRL library,
loads preprocessed stock data, and trains an RL agent for portfolio optimization.

Author: Your Name
Date: 2024-04-27
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from functools import reduce, partial
from itertools import cycle
from dataclasses import dataclass, field
import gym
from gym import spaces
from finrl import config, create_environment
from finrl.agents.stablebaselines3_models import DRLAgent
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.env.env_portfolio import PortfolioEnv
from finrl.plot import backtest_stats, backtest_plot

# Constants
PICKLE_DIR = "pickles"
MODEL_DIR = "models"
LOG_DIR = "logs"

def ensure_directories():
    """Ensure that necessary directories exist."""
    for directory in [MODEL_DIR, LOG_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

class RLConfig:
    """
    Configuration for the reinforcement learning environment and agent.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize with a configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): Configuration parameters.
        """
        self.config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with a default.

        Args:
            key (str): Configuration key.
            default (Any): Default value if key is not found.

        Returns:
            Any: Configuration value.
        """
        return self.config.get(key, default)

@dataclass
class PortfolioTradingEnv:
    """
    Custom Portfolio Trading Environment for Reinforcement Learning.
    """
    df: pd.DataFrame
    stock_dim: int
    hmax: float = 100
    initial_amount: float = 1000000
    buy_cost_pct: float = 1e-3
    sell_cost_pct: float = 1e-3
    state_space: int = field(init=False)
    action_space: int = field(init=False)
    tech_indicator_list: List[str] = field(default_factory=lambda: ['MACD', 'RSI', 'ADX'])

    def __post_init__(self):
        self.state_space = self.stock_dim * len(self.tech_indicator_list) + self.stock_dim * 2
        self.action_space = self.stock_dim

        self.env = PortfolioEnv(df=self.df,
                                stock_dim=self.stock_dim,
                                hmax=self.hmax,
                                initial_amount=self.initial_amount,
                                buy_cost_pct=self.buy_cost_pct,
                                sell_cost_pct=self.sell_cost_pct,
                                state_space=self.state_space,
                                action_space=self.action_space,
                                tech_indicator_list=self.tech_indicator_list)

    def get_env(self):
        """Get the FinRL Portfolio Environment."""
        return self.env

class RLTrainer:
    """
    Trains a Reinforcement Learning agent for portfolio optimization.
    """

    def __init__(self, config: RLConfig):
        """
        Initialize with RL configuration.

        Args:
            config (RLConfig): Configuration object.
        """
        self.config = config
        self.agent = None
        self.env = None

    def load_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Load and concatenate data from pickle files.

        Args:
            tickers (List[str]): List of stock ticker symbols.

        Returns:
            pd.DataFrame: Combined DataFrame of all stocks.
        """
        data_frames = []
        for ticker in tickers:
            file_path = os.path.join(PICKLE_DIR, f"{ticker}.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Pickle file for {ticker} not found.")
            with open(file_path, 'rb') as file:
                df = pickle.load(file)
                df['Ticker'] = ticker
                data_frames.append(df)
        combined_df = pd.concat(data_frames)
        return combined_df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data by engineering features and splitting.

        Args:
            df (pd.DataFrame): Combined stock data.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.config.get('tech_indicator_list', ['MACD', 'RSI', 'ADX']),
            use_turbulence=True,
            user_defined_feature=False
        )
        df = fe.preprocess_data(df)
        return df

    def setup_environment(self, df: pd.DataFrame) -> gym.Env:
        """
        Set up the RL environment.

        Args:
            df (pd.DataFrame): Preprocessed data.

        Returns:
            gym.Env: RL environment.
        """
        stock_dim = len(df.tic.unique())
        env_config = {
            'df': df,
            'stock_dim': stock_dim,
            'hmax': self.config.get('hmax', 100),
            'initial_amount': self.config.get('initial_amount', 1000000),
            'buy_cost_pct': self.config.get('buy_cost_pct', 1e-3),
            'sell_cost_pct': self.config.get('sell_cost_pct', 1e-3),
            'state_space': self.config.get('state_space', stock_dim * 10 + stock_dim * 2),
            'action_space': stock_dim,
            'tech_indicator_list': self.config.get('tech_indicator_list', ['MACD', 'RSI', 'ADX'])
        }
        trading_env = PortfolioTradingEnv(**env_config)
        self.env = trading_env.get_env()
        return self.env

    def train_agent(self, env: gym.Env) -> Any:
        """
        Train the RL agent using Stable Baselines3.

        Args:
            env (gym.Env): RL environment.

        Returns:
            Any: Trained agent.
        """
        self.agent = DRLAgent(env=env)
        model = self.agent.get_model(model_name=self.config.get('model_name', 'ppo'))
        trained_model = self.agent.train_model(model=model,
                                               tb_log_name=self.config.get('tb_log_name', 'ppo'),
                                               total_timesteps=self.config.get('total_timesteps', 10000))
        return trained_model

    def save_model(self, model: Any, model_name: str = 'ppo'):
        """
        Save the trained RL model.

        Args:
            model (Any): Trained model.
            model_name (str): Name of the model.
        """
        model_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    def run(self):
        """
        Execute the training pipeline.
        """
        tickers = self.config.get('tickers', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT'])
        df = self.load_data(tickers)
        df = self.preprocess_data(df)
        env = self.setup_environment(df)
        trained_model = self.train_agent(env)
        self.save_model(trained_model, self.config.get('model_name', 'ppo'))

def main():
    """Main function to execute the RL training pipeline."""
    ensure_directories()
    default_config = {
        'tech_indicator_list': ['MACD', 'RSI', 'ADX'],
        'hmax': 100,
        'initial_amount': 1000000,
        'buy_cost_pct': 1e-3,
        'sell_cost_pct': 1e-3,
        'model_name': 'ppo',
        'tb_log_name': 'ppo',
        'total_timesteps': 10000,
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT']
    }
    config = RLConfig(default_config)
    trainer = RLTrainer(config)
    trainer.run()

if __name__ == "__main__":
    main()
