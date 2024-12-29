# financial_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from metrics_computer import MetricsComputer, MetricsConfig
from observation.hht import mean_instantaneous_freq, mean_instantaneous_amp
import warnings

class PortfolioEnv(gym.Env):
    """
    A custom Gymnasium environment for portfolio optimization using reinforcement learning.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: Dict[str, Any], prices: pd.DataFrame, returns: pd.DataFrame):
        """
        Initialize the Portfolio Environment.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - window_size (int): Number of previous days to include in the observation.
                - num_assets (int): Number of assets in the portfolio.
                - metrics_list (List[Tuple[str, float]]): List of (metric_name, weight) tuples.
                - risk_free_rate (float): Risk-free rate for Sharpe and Sortino Ratios.
            prices (pd.DataFrame): Adjusted close prices.
            returns (pd.DataFrame): Daily percentage returns.
        """
        super(PortfolioEnv, self).__init__()
        
        # Extract configuration
        self.window_size = config.get('window_size', 20)
        self.num_assets = config.get('num_assets', 10)
        self.metrics_list = config.get('metrics_list', [])
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.compute_hht = config.get('compute_hht', False)
        self.extensive_metrics = config.get('extensive_metrics', False)
        
        # Data
        self.prices = prices.reset_index(drop=True)
        self.returns = returns.reset_index(drop=True)
        self.num_steps = len(self.prices) - self.window_size - 1  # Total steps
        
        # Metrics Computer
        self.metrics_computer = MetricsComputer(MetricsConfig(risk_free_rate=self.risk_free_rate))
        
        # Define action and observation space
        # Actions are the weights allocated to each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        

        obs_dict = {
            'prices': spaces.Box(low=0, high=np.inf, shape=(self.window_size, self.num_assets), dtype=np.float32),
            'returns': spaces.Box(low=-1, high=1, shape=(self.window_size, self.num_assets), dtype=np.float32),
        }

        if self.compute_hht:
            obs_dict["mean_instantaneous_freq"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)
            obs_dict["mean_instantaneous_amp"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)
            self.precompute_hht()

        if self.extensive_metrics:
            obs_dict['macd_hist'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            obs_dict['rsi'] = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
            

        # Observations are a dict containing:
        # - 'prices': past window_size days' prices
        # - 'returns': past window_size days' returns
        self.observation_space = spaces.Dict(obs_dict)
        
        # Initialize state variables
        self.current_step = self.window_size
        self.initial_balance = 100000  # Starting with $100,000
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.previous_weights = np.array([1.0 / self.num_assets] * self.num_assets)  # Equal weighting initially
        self.portfolio_history = [self.portfolio_value]  # Track portfolio value over time
        
        # Initialize equal-weighted cumulative return
        self.equal_weight_returns = self.returns.dot(np.ones(self.num_assets) / self.num_assets)
        self.equal_cumulative_return = (1 + self.equal_weight_returns).cumprod().iloc[-1] - 1
    
    def precompute_hht(self):
        from tqdm import tqdm  # Import tqdm for progress bar

        self.mean_instantaneous_freq_precomputed = np.zeros((len(self.prices), self.num_assets), dtype=np.float32)
        self.mean_instantaneous_amp_precomputed = np.zeros((len(self.prices), self.num_assets), dtype=np.float32)
        for step in tqdm(range(self.window_size, len(self.prices)), desc="Precomputing HHT"):
            datas = [self.prices.iloc[step - self.window_size:step].pct_change().dropna().values[:, i] for i in range(self.num_assets)]
            self.mean_instantaneous_freq_precomputed[step] = np.array([mean_instantaneous_freq(data) for data in datas], dtype=np.float32)
            self.mean_instantaneous_amp_precomputed[step] = np.array([mean_instantaneous_amp(data) for data in datas], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Args:
            seed (int, optional): Seed for the environment.
            options (Dict, optional): Additional options.
        
        Returns:
            observation (Dict[str, np.ndarray]): Initial observation.
        """
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.previous_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        self.portfolio_history = [self.portfolio_value]
        self.equal_cumulative_return = (1 + self.equal_weight_returns.iloc[self.current_step:self.current_step + 1]).cumprod().iloc[-1] - 1
        reward, self.computed_metrics = self.metrics_computer.compute(
            current_weights=self.previous_weights,
            observation=self._get_observation(pseudo=True),
            current_step_returns=self.returns.iloc[self.current_step],
            metrics_list=self.metrics_list
        )
        
        return self._get_observation(), {}
    
    def _get_observation(self, pseudo=False) -> Dict[str, np.ndarray]:
        """
        Get the current observation.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing past prices and returns.
        """
        obs_prices = self.prices.iloc[self.current_step - self.window_size:self.current_step].values
        obs_returns = self.returns.iloc[self.current_step - self.window_size:self.current_step].values

        return_dict = {
            'prices': obs_prices.astype(np.float32),
            'returns': obs_returns.astype(np.float32)
        }

        if pseudo:
            return return_dict

        if self.compute_hht:
            return_dict["mean_instantaneous_freq"] = self.mean_instantaneous_freq_precomputed[self.current_step]
            return_dict["mean_instantaneous_amp"] = self.mean_instantaneous_amp_precomputed[self.current_step]

        if self.extensive_metrics:
            return_dict["macd_hist"] = np.array([self.computed_metrics["macd_hist"]], dtype=np.float32)
            return_dict["rsi"] = np.array([self.computed_metrics["rsi"]], dtype=np.float32)

        return return_dict
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action (np.ndarray): Portfolio weights for each asset.
        
        Returns:
            observation (Dict[str, np.ndarray]): Next observation.
            reward (float): Reward obtained.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict[str, Any]): Additional information.
        """
        # Normalize the weights to sum to 1
        weights = action / np.sum(action) if np.sum(action) > 0 else self.previous_weights

        # Get the return for the current day
        daily_return = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(weights, daily_return)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate reward using MetricsComputer
        reward, self.computed_metrics = self.metrics_computer.compute(
            current_weights=weights,
            observation=self._get_observation(pseudo=True),
            current_step_returns=daily_return,
            metrics_list=self.metrics_list
        )
        
        # Move to the next step
        self.current_step += 1
        
        # Check if we've reached the end
        done = self.current_step >= len(self.prices) - 1
        truncated = False  # No truncation logic
        
        # Prepare the next observation
        if not done:
            observation = self._get_observation()
        else:
            observation = {
                'prices': np.zeros((self.window_size, self.num_assets), dtype=np.float32),
                'returns': np.zeros((self.window_size, self.num_assets), dtype=np.float32),
                'mean_instantaneous_freq': np.zeros((self.num_assets), dtype=np.float32),
                'mean_instantaneous_amp': np.zeros((self.num_assets), dtype=np.float32),
                'macd_hist': np.zeros((1), dtype=np.float32),
                'rsi': np.zeros((1), dtype=np.float32)
            }
        
        # Update previous weights
        self.previous_weights = weights.copy()
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'daily_return': portfolio_return,
            'weights': weights,
            'equal_weight_cumulative_return': self.equal_cumulative_return
        }
        
        return observation, reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        Plots the portfolio value over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio_history, label='RL Agent')
        plt.axhline(y=self.initial_balance * (1 + self.equal_cumulative_return), color='r', linestyle='--', label='Equal Weight Benchmark')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Portfolio Performance Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def close(self):
        """Close the environment."""
        pass
