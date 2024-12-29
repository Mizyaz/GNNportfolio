import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from experimental import optimal_portfolio, max_sharpe_ratio, min_variance, risk_parity
import yfinance as yf
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    """
    A Gymnasium environment for portfolio optimization using reinforcement learning.
    
    The environment implements a portfolio allocation strategy where the agent:
    - Observes the last 20 days of returns
    - Outputs portfolio weights for each asset
    - Receives rewards based on portfolio returns while considering risk
    """
    
    def __init__(self, price_data, start_idx, end_idx, risk_free_rate=0.0):
        super(PortfolioEnv, self).__init__()
        
        self.price_data = price_data
        self.returns_data = price_data.pct_change().fillna(0)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(price_data.columns)
        self.lookback_window = 20
        
        # Current position in the dataset
        self.current_step = self.start_idx + self.lookback_window
        
        # Portfolio tracking
        self.portfolio_value = 1000.0  # Initial investment
        self.current_weights = np.array([1/self.num_assets] * self.num_assets)
        self.weight_history = []
        self.portfolio_history = []
        
        # Define action space: portfolio weights for each asset
        # Sum of weights will be normalized to 1
        self.action_space = Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        # Define observation space: returns for last 20 days for each asset
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, self.num_assets),
            dtype=np.float32
        )

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.start_idx + self.lookback_window
        self.portfolio_value = 1000.0
        self.current_weights = np.array([1/self.num_assets] * self.num_assets)
        self.weight_history = []
        self.portfolio_history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        """Get the current observation (last 20 days of returns)."""
        obs_start = self.current_step - self.lookback_window
        obs_end = self.current_step
        observation = self.returns_data.iloc[obs_start:obs_end].values
        return observation

    def _calculate_reward(self, returns, weights):
        """
        Calculate reward based on portfolio return and risk.
        Implements a Sharpe ratio-like reward function.
        """
        portfolio_return = np.sum(returns * weights)
        
        # Calculate rolling window portfolio volatility
        rolling_returns = self.returns_data.iloc[self.current_step-30:self.current_step]
        portfolio_returns = np.sum(rolling_returns * weights, axis=1)
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio-like reward
        if portfolio_volatility == 0:
            reward = portfolio_return
        else:
            reward = (portfolio_return - self.risk_free_rate/252) / portfolio_volatility
            
        return reward

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Array of portfolio weights (will be normalized)
        
        Returns:
            observation: Next observation
            reward: Reward from the action
            done: Whether episode has ended
            truncated: Always False for this environment
            info: Additional information
        """
        # Ensure action is valid and normalize weights
        weights = np.array(action, dtype=np.float32)
        weights = np.clip(weights, 0, 1)
        weights_sum = np.sum(weights)
        
        # Handle the case where sum of weights is 0 or contains NaN
        if weights_sum <= 0 or np.isnan(weights_sum):
            weights = np.array([1/self.num_assets] * self.num_assets)
        else:
            weights = weights / weights_sum
        
        # Get returns for current step
        returns = self.returns_data.iloc[self.current_step].values
        
        # Calculate reward
        reward = self._calculate_reward(returns, weights)
        
        # Handle NaN reward
        if np.isnan(reward):
            reward = -1.0  # Penalize invalid actions
        
        # Update portfolio value
        portfolio_return = np.sum(returns * weights)
        if not np.isnan(portfolio_return):
            self.portfolio_value *= (1 + portfolio_return)
        
        # Store weights and portfolio value
        self.current_weights = weights
        self.weight_history.append(weights)
        self.portfolio_history.append(self.portfolio_value)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.end_idx
        
        # Get next observation
        if not done:
            next_observation = self._get_observation()
        else:
            # print current portfolio value
            print(f"Current portfolio value: {self.portfolio_value}")
            # print current weights and returns
            next_observation = np.zeros_like(self._get_observation())
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': weights,
            'returns': returns
        }
        
        return next_observation, reward, done, False, info

    def render(self):
        """Render the environment."""
        pass

    def get_portfolio_history(self):
        """Return the history of portfolio values."""
        return self.portfolio_history

    def get_weight_history(self):
        """Return the history of portfolio weights."""
        return self.weight_history

def train_rl_agent(data, start_train_idx, end_train_idx, start_test_idx, end_test_idx, risk_free_rate=0.0):
    """Trains a PPO agent for portfolio optimization."""

    env_train = lambda: PortfolioEnv(data.iloc[start_train_idx:end_train_idx], 0, len(data.iloc[start_train_idx:end_train_idx]) - 1, risk_free_rate)
    env_test = lambda: PortfolioEnv(data.iloc[start_test_idx:end_test_idx], 0, len(data.iloc[start_test_idx:end_test_idx]) - 1, risk_free_rate)
    env_train = DummyVecEnv([env_train])
    env_test = DummyVecEnv([env_test])

    model = PPO('MlpPolicy', env_train, verbose=1, learning_rate=0.0001, gamma=0.8)
    model.learn(total_timesteps=500000)  # Adjust training steps as needed

    # Evaluate on test data
    obs_test = env_test.reset()
    done = False
    portfolio_value = 1.0
    while not done:
        action, _ = model.predict(obs_test)
        obs_test, reward, done, _ = env_test.step(action)
        portfolio_value += reward

    return model, portfolio_value


def main():
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    start_date = '2021-10-26'
    end_date = '2023-10-26'
    risk_free_rate = 0.0
    initial_investment = 1000

    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if data.empty:
        print("No data downloaded. Check your tickers and date range.")
        return

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)
    equal_weights = np.array([1/num_assets] * num_assets)

    # Split data for training and testing RL agent
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Train RL agent
    model, _ = train_rl_agent(data, 0, split_index -1, split_index, len(data) - 1, risk_free_rate)

    # Backtesting all strategies
    portfolio_values = pd.DataFrame(index=test_data.index)

    # Max Sharpe
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    portfolio_values['Max Sharpe'] = (1 + test_data.pct_change().dropna() @ max_sharpe['x']).cumprod() * initial_investment
    
    # Min Variance
    min_var = min_variance(mean_returns, cov_matrix)
    portfolio_values['Min Variance'] = (1 + test_data.pct_change().dropna() @ min_var['x']).cumprod() * initial_investment

    # Risk Parity
    risk_parity_weights = risk_parity(cov_matrix)
    portfolio_values['Risk Parity'] = (1 + test_data.pct_change().dropna() @ risk_parity_weights).cumprod() * initial_investment

    # Equal Weight
    portfolio_values['Equal Weight'] = (1 + test_data.pct_change().dropna() @ equal_weights).cumprod() * initial_investment

    # RL Agent
    env_test = PortfolioEnv(test_data, 0, len(test_data) - 1, risk_free_rate)
    obs, _ = env_test.reset()
    rl_portfolio_values = [initial_investment]
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env_test.step(action)
        rl_portfolio_values.append(rl_portfolio_values[-1] * (1 + rewards))

    rl_agent_values = rl_portfolio_values[1:]
    required_length = len(portfolio_values)  # Length of the portfolio_values DataFrame
    if len(rl_agent_values) < required_length:
        # Pad the beginning with initial investment
        padding = [initial_investment] * (required_length - len(rl_agent_values))
        portfolio_values['RL Agent'] = padding + rl_agent_values
    else:
        portfolio_values['RL Agent'] = rl_agent_values
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time (Comparison)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend(portfolio_values.columns)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()