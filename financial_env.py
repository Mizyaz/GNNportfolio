import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from black_litterman import BlackLittermanModel
from financial_params import FinancialMetrics, FinancialParameters

class ObservationType(Enum):
    """Types of observations that can be included in the state"""
    PRICES = "prices"
    RETURNS = "returns"
    TECHNICAL = "technical"
    TREND = "trend"
    BLACK_LITTERMAN = "black_litterman"
    DIVERSIFICATION = "diversification"

class RewardType(Enum):
    """Types of reward components"""
    SHARPE = "sharpe"
    SORTINO = "sortino"
    DIVERSIFICATION = "diversification"
    TURNOVER = "turnover"
    TRACKING_ERROR = "tracking_error"
    BLACK_LITTERMAN = "black_litterman"

@dataclass
class EnvironmentConfig:
    """Configuration for the financial environment"""
    window_size: int = 20
    prediction_days: int = 5
    num_assets: int = 10
    observation_types: List[ObservationType] = None
    reward_types: List[RewardType] = None
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.02
    target_volatility: float = 0.15
    regularization_factor: float = 1e-6
    max_steps: Optional[int] = None
    
    def __post_init__(self):
        if self.observation_types is None:
            self.observation_types = list(ObservationType)
        if self.reward_types is None:
            self.reward_types = list(RewardType)

class ObservationFactory:
    """Factory for creating observation components"""
    
    @staticmethod
    def create_observation_space(config: EnvironmentConfig) -> Dict[str, gym.Space]:
        """Create observation spaces for each component"""
        spaces = {}
        
        for obs_type in config.observation_types:
            if obs_type == ObservationType.PRICES:
                spaces[obs_type.value] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(config.window_size, config.num_assets)
                )
            elif obs_type == ObservationType.RETURNS:
                spaces[obs_type.value] = gym.spaces.Box(
                    low=-1, high=1, 
                    shape=(config.window_size, config.num_assets)
                )
            elif obs_type == ObservationType.TECHNICAL:
                # RSI, BB, Volatility
                spaces[obs_type.value] = gym.spaces.Box(
                    low=0, high=1, 
                    shape=(3, config.num_assets)
                )
            elif obs_type == ObservationType.TREND:
                # MA ratios, Momentum, Trend strength
                spaces[obs_type.value] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(3, config.num_assets)
                )
            elif obs_type == ObservationType.BLACK_LITTERMAN:
                # Expected returns, Covariance
                spaces[obs_type.value] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(2, config.num_assets)
                )
            elif obs_type == ObservationType.DIVERSIFICATION:
                # Correlation, Volatility contribution
                spaces[obs_type.value] = gym.spaces.Box(
                    low=-1, high=1, 
                    shape=(2, config.num_assets)
                )
                
        return spaces

class RewardFactory:
    """Factory for creating reward components"""
    
    @staticmethod
    def create_reward_functions(config: EnvironmentConfig) -> Dict[str, Callable]:
        """Create reward functions with better scaling"""
        functions = {}
        
        for reward_type in config.reward_types:
            if reward_type == RewardType.SHARPE:
                functions[reward_type.value] = lambda returns, vol: np.clip(
                    (np.mean(returns) - config.risk_free_rate) / (vol + 1e-6),
                    -1, 1
                )
            elif reward_type == RewardType.SORTINO:
                functions[reward_type.value] = lambda returns: np.clip(
                    RewardFactory._calculate_sortino_ratio(
                        returns, config.risk_free_rate
                    ),
                    -1, 1
                )
            elif reward_type == RewardType.DIVERSIFICATION:
                functions[reward_type.value] = lambda weights: np.clip(
                    -np.sum(weights ** 2) + 1,  # Shifted to [-1, 1] range
                    -1, 1
                )
            elif reward_type == RewardType.TURNOVER:
                functions[reward_type.value] = lambda weights: np.clip(
                    -np.sum(np.abs(weights - np.ones(config.num_assets)/config.num_assets)) * config.transaction_cost,
                    -1, 1
                )
            elif reward_type == RewardType.TRACKING_ERROR:
                functions[reward_type.value] = lambda weights: np.clip(
                    -np.sum((weights - 1/config.num_assets) ** 2),
                    -1, 1
                )
            elif reward_type == RewardType.BLACK_LITTERMAN:
                functions[reward_type.value] = lambda weights, bl_returns: np.clip(
                    np.dot(weights, bl_returns),
                    -1, 1
                )
                
        return functions
    
    @staticmethod
    def _calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate
        downside_returns = np.where(returns < 0, returns, 0)
        downside_std = np.std(downside_returns) + 1e-6
        return np.mean(excess_returns) / downside_std

class FinancialEnv(gym.Env):
    """
    A modular financial environment for portfolio optimization
    
    The environment uses a factory pattern to create observation and reward components
    based on the provided configuration. It supports multiple observation types and
    reward components that can be combined flexibly.
    """
    
    def __init__(self, config: EnvironmentConfig, full_prices: np.ndarray, full_returns: np.ndarray):
        """Initialize environment with full data"""
        self.config = config
        self.full_prices = full_prices
        self.full_returns = full_returns
        
        # Set max_steps based on data length if not specified
        if self.config.max_steps is None:
            self.config.max_steps = (len(full_returns) - self.config.window_size) // self.config.prediction_days
        
        # Create action and observation spaces
        self.action_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(config.num_assets,), 
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict(
            ObservationFactory.create_observation_space(config)
        )
        
        # Initialize components
        self.bl_model = BlackLittermanModel(
            num_assets=config.num_assets,
            window_size=config.window_size
        )
        self.financial_metrics = FinancialMetrics()
        self.reward_functions = RewardFactory.create_reward_functions(config)
        
        # Initialize state variables
        self.current_step = 0
        self.current_idx = 0
        self.returns_history = None
        self.prices_history = None
        self.previous_weights = None
        
        # Add episode tracking
        self.episode_returns = []
        self.episode_start_idx = None
        self.episode_end_idx = None

    def reset(self, seed=None, options=None):
        """Reset environment to start new episode"""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.episode_returns = []
        
        # Randomly select starting point that allows for a full episode
        max_start = len(self.full_returns) - (self.config.window_size + self.config.max_steps * self.config.prediction_days)
        self.current_idx = self.np_random.integers(self.config.window_size, max_start) if max_start > self.config.window_size else self.config.window_size
        
        self.episode_start_idx = self.current_idx
        self.current_step = 0
        
        # Initialize histories with actual data
        self.returns_history = self.full_returns[
            self.current_idx - self.config.window_size:self.current_idx
        ]
        self.prices_history = self.full_prices[
            self.current_idx - self.config.window_size:self.current_idx
        ]
        
        self.previous_weights = np.ones(self.config.num_assets) / self.config.num_assets
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray):
        """Execute one step, applying weights for prediction_days"""
        try:
            weights = self._normalize_weights(action)
            
            # Calculate returns for the prediction period
            k_day_returns = []
            k_day_prices = []
            
            for k in range(self.config.prediction_days):
                day_idx = self.current_idx + k
                if day_idx >= len(self.full_returns):
                    break
                    
                daily_return = np.dot(weights, self.full_returns[day_idx])
                k_day_returns.append(daily_return)
                k_day_prices.append(self.full_prices[day_idx])
                
            # Store returns for episode tracking
            self.episode_returns.extend(k_day_returns)
            
            # Update current index and step counter
            self.current_idx += self.config.prediction_days
            self.current_step += 1
            
            # Update histories
            if self.current_idx < len(self.full_returns):
                self.returns_history = self.full_returns[
                    self.current_idx - self.config.window_size:self.current_idx
                ]
                self.prices_history = self.full_prices[
                    self.current_idx - self.config.window_size:self.current_idx
                ]
            
            # Calculate reward
            reward = self._calculate_reward(weights)
            
            # Check if episode is done
            done = (self.current_idx >= len(self.full_returns) - self.config.prediction_days)
            
            if done:
                self.episode_end_idx = self.current_idx
            
            # Calculate episode metrics
            episode_metrics = self._get_episode_metrics() if done else {}
            
            info = {
                'portfolio_return': float(np.mean(k_day_returns)) if k_day_returns else 0.0,
                'weights': weights.tolist(),
                'reward_components': self._get_reward_components(weights),
                'k_day_returns': k_day_returns,
                'current_idx': self.current_idx,
                'episode_start': self.episode_start_idx,
                'episode_end': self.episode_end_idx if done else None,
                'current_step': self.current_step,
                **episode_metrics
            }
            
            self.previous_weights = weights.copy()
            
            return self._get_observation(), reward, done, False, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), -1.0, True, False, {}
            
    def _get_episode_metrics(self) -> dict:
        """Calculate episode-level metrics"""
        if not self.episode_returns:
            return {}
            
        episode_returns = np.array(self.episode_returns)
        cumulative_return = np.prod(1 + episode_returns) - 1
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        sharpe = mean_return / (std_return + 1e-6)
        
        return {
            'episode_cumulative_return': float(cumulative_return),
            'episode_mean_return': float(mean_return),
            'episode_sharpe': float(sharpe),
            'episode_length': len(episode_returns),
            'episode_volatility': float(std_return)
        }
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1 with bounds"""
        weights = np.clip(weights, 0, 1)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            return weights / weight_sum
        return np.ones(self.config.num_assets) / self.config.num_assets
    
    def _calculate_portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio return"""
        return np.dot(weights, self.returns_history[-1])
    
    def _update_histories(self, portfolio_return: float):
        """Update histories with proper safeguards"""
        try:
            # Ensure arrays are properly initialized
            if self.returns_history is None or self.prices_history is None:
                self.reset()
                
            # Ensure minimum history length
            if len(self.returns_history) < self.config.window_size:
                pad_length = self.config.window_size - len(self.returns_history)
                self.returns_history = np.pad(
                    self.returns_history,
                    ((pad_length, 0), (0, 0)),
                    mode='constant',
                    constant_values=0.0
                )
                self.prices_history = np.pad(
                    self.prices_history,
                    ((pad_length, 0), (0, 0)),
                    mode='constant',
                    constant_values=1.0
                )
            
            # Update returns history
            self.returns_history = np.roll(self.returns_history, -1, axis=0)
            self.returns_history[-1] = np.clip(portfolio_return, -1, 1)  # Clip extreme returns
            
            # Update prices history
            self.prices_history = np.roll(self.prices_history, -1, axis=0)
            self.prices_history[-1] = self.prices_history[-2] * (1 + self.returns_history[-1])
            
        except Exception as e:
            print(f"Warning: Error updating histories: {e}")
            self.reset()  # Reset if something goes wrong
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation based on configured observation types"""
        observation = {}
        
        for obs_type in self.config.observation_types:
            if obs_type == ObservationType.PRICES:
                observation[obs_type.value] = self._normalize_prices(self.prices_history)
            elif obs_type == ObservationType.RETURNS:
                observation[obs_type.value] = self.returns_history
            elif obs_type == ObservationType.TECHNICAL:
                observation[obs_type.value] = self._get_technical_indicators()
            elif obs_type == ObservationType.TREND:
                observation[obs_type.value] = self._get_trend_indicators()
            elif obs_type == ObservationType.BLACK_LITTERMAN:
                observation[obs_type.value] = self._get_black_litterman_params()
            elif obs_type == ObservationType.DIVERSIFICATION:
                observation[obs_type.value] = self._get_diversification_metrics()
        
        return observation
    
    def _calculate_reward(self, weights: np.ndarray) -> float:
        """Calculate combined reward with clipping"""
        components = self._get_reward_components(weights)
        # Clip individual components
        clipped_components = {
            k: np.clip(v, -1, 1) for k, v in components.items()
        }
        # Average clipped components
        return np.mean(list(clipped_components.values()))
    
    def _get_reward_components(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate reward components with proper handling"""
        components = {}
        
        for reward_type, func in self.reward_functions.items():
            try:
                if reward_type == RewardType.SHARPE.value:
                    if len(self.returns_history) < self.config.window_size:
                        components[reward_type] = 0.0
                        continue
                        
                    returns = self.returns_history[-self.config.window_size:]
                    vol = np.std(returns) + 1e-6
                    components[reward_type] = func(returns, vol)
                    
                elif reward_type == RewardType.BLACK_LITTERMAN.value:
                    if len(self.returns_history) < self.config.window_size:
                        components[reward_type] = 0.0
                        continue
                        
                    returns, _ = self.bl_model.optimize(
                        self.returns_history[-self.config.window_size:]
                    )
                    components[reward_type] = func(weights, returns)
                    
                else:
                    components[reward_type] = func(weights)
                
                # Clip extreme values
                components[reward_type] = np.clip(components[reward_type], -1, 1)
                
            except Exception as e:
                components[reward_type] = 0.0
        
        return components
    
    # Helper methods for observation components...
    def _normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Normalize prices to recent window"""
        return prices / np.mean(prices)
    
    def _get_technical_indicators(self) -> np.ndarray:
        """Get technical indicators with safeguards"""
        try:
            indicators = self.financial_metrics.calculate_technical_indicators(
                self.prices_history[-self.config.window_size:]
            )
        except Exception as e:
            print(f"Warning: Error calculating technical indicators: {e}")
            indicators = {
                'rsi': np.zeros(self.config.num_assets),
                'bb_position': np.zeros(self.config.num_assets),
                'volatility': np.zeros(self.config.num_assets)
            }
        
        # Safe normalization
        return np.vstack([
            indicators['rsi'] / 100,  # Normalize to [0, 1]
            (indicators['bb_position'] + 1) / 2,  # Normalize to [0, 1]
            indicators['volatility'][0] / np.max(indicators['volatility'])  # Normalize
        ])

    def _get_trend_indicators(self) -> np.ndarray:
        """Get trend indicators with proper window handling"""
        try:
            # Ensure we have enough history
            if len(self.prices_history) < self.config.window_size:
                return np.zeros((3, self.config.num_assets))
            
            # Use only complete windows
            window_prices = self.prices_history[-self.config.window_size:]
            
            trend_metrics = self.financial_metrics.calculate_trend_metrics(window_prices)
            
            # Safe normalization with clipping
            ma_ratios = np.clip(trend_metrics['ma_ratios'][0], -5, 5)
            momentum = np.clip(trend_metrics['momentum'][0], -1, 1)
            trend_strength = np.clip(trend_metrics['trend_strength'], 0, 1)
            
            return np.vstack([ma_ratios, momentum, trend_strength])
            
        except Exception as e:
            return np.zeros((3, self.config.num_assets))
    
    def _get_black_litterman_params(self) -> np.ndarray:
        """Get Black-Litterman parameters with robust handling"""
        try:
            # Ensure we have enough history
            if len(self.returns_history) < self.config.window_size:
                return np.zeros((2, self.config.num_assets))
            
            # Add small regularization to prevent singular matrix
            window_returns = self.returns_history[-self.config.window_size:]
            window_returns += np.random.normal(0, 1e-8, window_returns.shape)
            
            returns, cov = self.bl_model.optimize(window_returns)
            
            # Ensure finite values and proper scaling
            returns = np.clip(returns, -0.1, 0.1)
            vol = np.clip(np.diag(cov), 0, 0.1)
            
            return np.vstack([returns, vol])
            
        except Exception as e:
            return np.zeros((2, self.config.num_assets))
    
    def _get_diversification_metrics(self) -> np.ndarray:
        """Get diversification metrics with safeguards"""
        try:
            # Use proper window size
            window_returns = self.returns_history[-self.config.window_size:]
            
            # Calculate correlation matrix safely
            corr_matrix = np.corrcoef(window_returns.T)
            if np.isnan(corr_matrix).any():
                corr_matrix = np.eye(self.config.num_assets)
                
            metrics = self.financial_metrics.calculate_diversification_metrics(
                window_returns
            )
            
            # Ensure proper shape and scaling
            correlations = np.clip(np.diag(corr_matrix), -1, 1)
            penalty = np.clip(
                metrics.get('correlation_penalty', 0) * np.ones(self.config.num_assets),
                -1, 1
            )
            
            return np.vstack([correlations, penalty])
            
        except Exception as e:
            print(f"Warning: Error calculating diversification metrics: {e}")
            return np.zeros((2, self.config.num_assets)) 