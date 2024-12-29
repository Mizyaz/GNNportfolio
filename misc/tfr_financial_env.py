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

class SpectralPortfolioEnv(gym.Env):
    """
    A refined Gymnasium environment for portfolio optimization using mel spectrograms.
    
    Key modifications:
    - Aligned spectrogram dimensions with input data structure
    - Optimized window parameters for financial time series
    - Enhanced numerical stability in frequency domain transformations
    """
    
    def __init__(self, price_data, start_idx, end_idx, risk_free_rate=0.0):
        super(SpectralPortfolioEnv, self).__init__()
        
        # Data initialization
        self.price_data = price_data
        self.returns_data = price_data.pct_change().fillna(0)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(price_data.columns)
        
        # Spectrogram parameters - adjusted for financial data
        self.sample_rate = 252  # Trading days per year
        self.n_fft = 32        # Length of FFT window
        self.hop_length = 16   # Number of samples between successive frames
        self.n_mels = 32       # Number of mel bands
        self.lookback_window = 252  # One year of trading data
        
        # Environment state
        self.current_step = self.start_idx + self.lookback_window
        
        # Portfolio state
        self.portfolio_value = 1000.0
        self.current_weights = np.array([1/self.num_assets] * self.num_assets)
        self.weight_history = []
        self.portfolio_history = []
        
        # Action space: portfolio weights
        self.action_space = Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        # Observation space: mel spectrograms for each asset
        # Shape: (n_mels, time_steps, num_assets)
        n_frames = 1 + (self.lookback_window - self.n_fft) // self.hop_length
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_mels, n_frames, self.num_assets),
            dtype=np.float32
        )

    def _compute_mel_spectrogram(self, returns_window):
        """
        Compute mel spectrogram with refined parameters for financial time series.
        
        Args:
            returns_window: Array of asset returns
            
        Returns:
            mel_spectrogram: 2D array of shape (n_mels, time_steps)
        """
        # Normalize returns
        normalized_returns = (returns_window - np.mean(returns_window)) / (np.std(returns_window) + 1e-8)
        
        # Compute STFT
        stft = librosa.stft(
            normalized_returns,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=False
        )
        
        # Convert to power spectrogram
        power_spec = np.abs(stft) ** 2
        
        # Create mel filterbank
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )
        
        # Apply mel filterbank
        mel_spec = np.dot(mel_basis, power_spec)
        
        # Log-scale and normalize
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec

    def _get_observation(self):
        """
        Construct the current observation as mel spectrograms for all assets.
        
        Returns:
            observation: Array of shape (n_mels, time_steps, num_assets)
        """
        obs_start = self.current_step - self.lookback_window
        obs_end = self.current_step
        
        # Initialize observation array
        n_frames = 1 + (self.lookback_window - self.n_fft) // self.hop_length
        observation = np.zeros((self.n_mels, n_frames, self.num_assets))
        
        # Compute mel spectrograms for each asset
        for i, asset in enumerate(self.returns_data.columns):
            returns = self.returns_data[asset].iloc[obs_start:obs_end].values
            observation[:, :, i] = self._compute_mel_spectrogram(returns)
        
        return observation.astype(np.float32)

    def _calculate_reward(self, returns, weights):
        """
        Calculate reward using risk-adjusted returns.
        """
        portfolio_return = np.sum(returns * weights)
        
        # Calculate rolling window portfolio volatility
        rolling_returns = self.returns_data.iloc[self.current_step-30:self.current_step]
        portfolio_returns = np.sum(rolling_returns * weights, axis=1)
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        if portfolio_volatility == 0:
            reward = portfolio_return
        else:
            reward = (portfolio_return - self.risk_free_rate/252) / portfolio_volatility
            
        return reward

    def step(self, action):
        """Execute one time step within the environment."""
        # Normalize weights
        weights = np.array(action, dtype=np.float32)
        weights = np.clip(weights, 0, 1)
        weights_sum = np.sum(weights)
        
        if weights_sum <= 0 or np.isnan(weights_sum):
            weights = np.array([1/self.num_assets] * self.num_assets)
        else:
            weights = weights / weights_sum
        
        # Get current returns and calculate reward
        returns = self.returns_data.iloc[self.current_step].values
        reward = self._calculate_reward(returns, weights)
        
        if np.isnan(reward):
            reward = -1.0
        
        # Update portfolio value
        portfolio_return = np.sum(returns * weights)
        if not np.isnan(portfolio_return):
            self.portfolio_value *= (1 + portfolio_return)
        
        # Store history
        self.current_weights = weights
        self.weight_history.append(weights)
        self.portfolio_history.append(self.portfolio_value)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.end_idx
        
        if not done:
            next_observation = self._get_observation()
        else:
            next_observation = np.zeros_like(self._get_observation())
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': weights,
            'returns': returns
        }
        
        return next_observation, reward, done, False, info

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.start_idx + self.lookback_window
        self.portfolio_value = 1000.0
        self.current_weights = np.array([1/self.num_assets] * self.num_assets)
        self.weight_history = []
        self.portfolio_history = []
        
        return self._get_observation(), {}

