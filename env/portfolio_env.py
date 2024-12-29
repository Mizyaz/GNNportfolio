# portfolio_env.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import plotext as plt
from typing import Dict, Tuple, List
from env.config import PortfolioConfig
from env.feature_computer import FeatureComputer
from env.risk_metrics import RiskMetricsComputer
from env.reward import Reward, RewardConfig, compute_sharpe, compute_sortino, compute_max_drawdown_reward, compute_calmar_ratio_reward, compute_var, compute_cvar

class PortfolioEnv(gym.Env):
    """Enhanced Portfolio Environment with a dictionary-based observation space."""
    
    def __init__(self,
                 config: PortfolioConfig,
                 prices: pd.DataFrame,
                 returns: pd.DataFrame,
                 high_prices: pd.DataFrame = None,
                 low_prices: pd.DataFrame = None,
                 volumes: pd.DataFrame = None,
                 existing_results: Dict[str, Tuple[float, List[float]]] = None):
        super().__init__()
        
        self.config = config
        self.feature_computer = FeatureComputer(config.tech_indicator_config)
        self.risk_computer = RiskMetricsComputer()
        
        self.prices = prices.copy()
        self.returns = returns.copy()
        self.high_prices = high_prices.copy() if high_prices is not None else None
        self.low_prices = low_prices.copy() if low_prices is not None else None
        self.volumes = volumes.copy() if volumes is not None else None
        self.colors = ["green", "orange", "purple"]

        self.existing_results = existing_results

        self.equal_weight_baseline_final_value, self.equal_weight_baseline_portfolio_history = run_equal_weight_baseline(prices, returns, config)
        
        # Validate data shape
        if len(self.prices.columns) != self.config.num_assets:
            raise ValueError("Number of assets in prices does not match config.num_assets.")
        
        self.current_step = self.config.window_size
        self.portfolio_value = self.config.initial_balance
        self.previous_weights = np.ones(self.config.num_assets) / self.config.num_assets
        self.portfolio_history = [self.portfolio_value]
        
        # For logging
        self.episode_count = 0
        self.global_portfolio_values = []
        self.all_portfolio_histories = []
        
        # Precompute features
        self.features = {asset: {} for asset in range(self.config.num_assets)}
        if self.config.use_technical_indicators:
            self._precompute_features()
            

        # Initialize Reward System
        if self.config.reward_config is None:
            # Define a default reward configuration if none is provided
            reward_config = RewardConfig(
                rewards=[
                    (compute_sharpe, 1.0, {"window": 20, "risk_free_rate": self.config.risk_free_rate}),
                    (compute_sortino, 0.5, {"window": 20, "risk_free_rate": self.config.risk_free_rate}),
                    (compute_max_drawdown_reward, 0.3, {"lookback": 50}),
                    (compute_calmar_ratio_reward, 0.2, {"window": 252, "risk_free_rate": self.config.risk_free_rate}),
                    (compute_var, -0.1, {"window": 20, "confidence": 0.95}),
                    (compute_cvar, -0.2, {"window": 20, "confidence": 0.95}),
                ]
            )
        else:
            reward_config = self.config.reward_config
        self.reward_system = Reward(reward_config)
        self.episode_rewards_global = []
        self.episode_rewards_global_mean = []
        self.episode_rewards_global_std = []
        self.episode_rewards_global_sum = []


        # Setup observation and action spaces
        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.config.num_assets,), dtype=np.float32
        )
    
    def _build_observation_space(self) -> spaces.Dict:
        """Construct the dictionary-based observation space."""
        # Basic shapes
        obs_spaces = {
            "prices": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.config.window_size, self.config.num_assets),
                dtype=np.float32
            ),
            "returns": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.config.window_size, self.config.num_assets),
                dtype=np.float32
            ),
            "weights": spaces.Box(
                low=0, high=1,
                shape=(self.config.num_assets,),
                dtype=np.float32
            ),
            "portfolio_value": spaces.Box(
                low=0, high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
        }
        
        if self.config.use_technical_indicators:
            # Count total indicators
            n_indicators = 0
            n_indicators += len(self.config.tech_indicator_config.sma_periods)
            n_indicators += len(self.config.tech_indicator_config.ema_periods)
            # RSI, MACD(3), BB(3)
            n_indicators += 1 + 3 + 3
            if self.high_prices is not None and self.low_prices is not None:
                # ATR, ADX, Aroon(2), CCI, MFI
                n_indicators += 1 + 1 + 2 + 1 + 1
            if self.volumes is not None:
                # OBV, volume SMA (if we had it)
                n_indicators += 2  # Adjust if you add more volume indicators
                
            obs_spaces["technical_indicators"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.config.num_assets, n_indicators),
                dtype=np.float32
            )
        
        if self.config.use_correlation_features:
            n_corr = self.config.num_assets * (self.config.num_assets - 1) // 2
            obs_spaces["correlation"] = spaces.Box(
                low=-1, high=1,
                shape=(n_corr,),
                dtype=np.float32
            )
        
        if self.config.use_risk_metrics:
            # 7: volatility, sharpe, sortino, var, cvar, max_drawdown, calmar
            obs_spaces["risk_metrics"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,),
                dtype=np.float32
            )
        
        if self.config.use_time_freq and self.config.tech_indicator_config.use_time_freq:
            obs_spaces["time_freq"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.config.num_assets, 4),
                dtype=np.float32
            )
        
        return spaces.Dict(obs_spaces)
    
    def _precompute_features(self):
        """Precompute all technical indicators for every asset."""
        for asset in range(self.config.num_assets):
            asset_features = {}
            p = self.prices.iloc[:, asset].values
            
            # Basic indicators
            ma = self.feature_computer.compute_moving_averages(p)
            osc = self.feature_computer.compute_oscillators(p)
            vol = self.feature_computer.compute_volatility(p)
            asset_features.update(ma)
            asset_features.update(osc)
            asset_features.update(vol)
            
            # If OHLC available
            if self.high_prices is not None and self.low_prices is not None:
                high = self.high_prices.iloc[:, asset].values
                low = self.low_prices.iloc[:, asset].values
                v = self.volumes.iloc[:, asset].values if self.volumes is not None else None
                oh_osc = self.feature_computer.compute_oscillators(p, high, low, v)
                oh_vol = self.feature_computer.compute_volatility(p, high, low)
                momentum = self.feature_computer.compute_momentum(p, high, low)
                asset_features.update(oh_osc)
                asset_features.update(oh_vol)
                asset_features.update(momentum)
            
            # Time frequency
            if self.config.use_time_freq and self.config.tech_indicator_config.use_time_freq:
                asset_features['time_freq'] = self.feature_computer.compute_time_frequency_features(p)
            
            self.features[asset] = asset_features
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.config.window_size
        self.portfolio_value = self.config.initial_balance
        self.previous_weights = np.ones(self.config.num_assets) / self.config.num_assets
        self.portfolio_history = [self.portfolio_value]
        self.episode_rewards = []
        return self._get_observation(), {}
    
    def step(self, action):
        # Normalize action
        weights = action / (np.sum(action) + 1e-8)
        
        # Transaction costs
        turnover = np.sum(np.abs(weights - self.previous_weights))
        transaction_costs = turnover * self.config.transaction_cost
        
        # Portfolio return
        current_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(weights, current_returns) - transaction_costs
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Compute reward
        reward = self.reward_system.compute_reward(self)
        
        # Update state
        self.previous_weights = weights
        self.current_step += 1
        
        # Check done
        done = (self.current_step >= len(self.prices) - 1)
        
        # If done, do some final logging / plotting
        if done:
            self.process_episode()

        
        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "transaction_costs": transaction_costs,
            "weights": weights
        }

        self.episode_rewards.append(reward)
        return obs, reward, done, False, info
    
    def process_episode(self):            
        self.episode_count += 1
        self.global_portfolio_values.append(self.portfolio_value)
        self.all_portfolio_histories.append(self.portfolio_history)
        
        if self.config.use_per_episode_plot:
            # Plot with plotext
            plt.clear_figure()
            plt.plot(self.portfolio_history, label="Trained Agent")
            plt.plot(self.equal_weight_baseline_portfolio_history, color="red", label="Equal Weight Baseline")
            """for i, key in enumerate(self.existing_results.keys()):
            plt.plot(self.existing_results[key][1], label=key, color=self.colors[i])
            """
            plt.title(f"Episode {self.episode_count} - Portfolio Value Over Time: {self.portfolio_value:.2f}")
            plt.show()
        
        print("Episode", self.episode_count, "current PV:", np.round(self.portfolio_value, 2), end="\t")
        print("Equal Weight Baseline:", np.round((self.equal_weight_baseline_final_value+1)*self.config.initial_balance, 2), end="\t")
        for i, key in enumerate(self.existing_results.keys()):
            print(key, np.round((self.existing_results[key][0]+1)*self.config.initial_balance, 2), end="\t")
        print()

        print("Episode Rewards:", np.sum(self.episode_rewards).round(2), "Mean Reward:", np.mean(self.episode_rewards).round(2), "Std Reward:", np.std(self.episode_rewards).round(2))
        self.episode_rewards_global.append(self.episode_rewards)
        self.episode_rewards_global_mean.append(np.mean(self.episode_rewards))
        self.episode_rewards_global_std.append(np.std(self.episode_rewards))
        self.episode_rewards_global_sum.append(np.sum(self.episode_rewards))
        print("Episode Rewards Global Mean:", np.mean(self.episode_rewards_global_sum).round(2), "Std:", np.std(self.episode_rewards_global_sum).round(2), "Max:", np.max(self.episode_rewards_global_sum).round(2), "Min:", np.min(self.episode_rewards_global_sum).round(2))
        
        if self.episode_count % 10 == 0:
            plt.clear_figure()
            plt.hline((self.equal_weight_baseline_final_value + 1)*self.config.initial_balance, color="red")
            for i, key in enumerate(self.existing_results.keys()):
                plt.hline((self.existing_results[key][0]+1)*self.config.initial_balance, color=self.colors[i])
            plt.plot(self.global_portfolio_values, color="red")
            mean_val = np.mean(self.global_portfolio_values)
            plt.title(f"Overall Performance up to Episode {self.episode_count} (Mean PV: {mean_val:.2f})")
            plt.show()
            plt.clear_figure()
            plt.plot(self.episode_rewards_global_sum)
            plt.show()
    
    def _get_observation(self):
        start_idx = self.current_step - self.config.window_size
        end_idx = self.current_step
        
        obs = {
            "prices": self.prices.iloc[start_idx:end_idx].values.astype(np.float32),
            "returns": self.returns.iloc[start_idx:end_idx].values.astype(np.float32),
            "weights": self.previous_weights.astype(np.float32),
            "portfolio_value": np.array([self.portfolio_value], dtype=np.float32),
        }
        
        # Technical indicators
        if self.config.use_technical_indicators:
            ti_matrix = []
            for asset in range(self.config.num_assets):
                asset_feat = self.features[asset]
                row_feats = []
                # SMA
                for period in self.config.tech_indicator_config.sma_periods:
                    val = asset_feat.get(f'sma_{period}', np.zeros_like(self.prices.iloc[:, 0]))[self.current_step] \
                          if self.current_step < len(self.prices) else 0.0
                    row_feats.append(val if not np.isnan(val) else 0.0)
                # EMA
                for period in self.config.tech_indicator_config.ema_periods:
                    val = asset_feat.get(f'ema_{period}', np.zeros_like(self.prices.iloc[:, 0]))[self.current_step] \
                          if self.current_step < len(self.prices) else 0.0
                    row_feats.append(val if not np.isnan(val) else 0.0)
                # RSI, MACD(3), BB(3)
                for ind in ["rsi", "macd", "macd_signal", "macd_hist",
                            "bb_upper", "bb_middle", "bb_lower"]:
                    arr = asset_feat.get(ind, np.zeros_like(self.prices.iloc[:, 0]))
                    val = arr[self.current_step] if self.current_step < len(arr) else 0.0
                    row_feats.append(val if not np.isnan(val) else 0.0)
                # OHLC indicators
                if self.high_prices is not None and self.low_prices is not None:
                    for ind in ["atr", "adx", "aroon_up", "aroon_down", "cci", "mfi"]:
                        arr = asset_feat.get(ind, np.zeros_like(self.prices.iloc[:, 0]))
                        val = arr[self.current_step] if self.current_step < len(arr) else 0.0
                        row_feats.append(val if not np.isnan(val) else 0.0)
                # Volume-based (obv, volume_sma) if you add them
                if self.volumes is not None:
                    for ind in ["obv", "volume_sma"]:
                        arr = asset_feat.get(ind, np.zeros_like(self.prices.iloc[:, 0]))
                        val = arr[self.current_step] if self.current_step < len(arr) else 0.0
                        row_feats.append(val if not np.isnan(val) else 0.0)
                ti_matrix.append(row_feats)
            
            obs["technical_indicators"] = np.array(ti_matrix, dtype=np.float32)
        
        if self.config.use_correlation_features:
            # Correlation from last window
            corr_df = self.returns.iloc[start_idx:end_idx].corr()
            corr_vals = corr_df.values[np.triu_indices_from(corr_df.values, k=1)]
            obs["correlation"] = corr_vals.astype(np.float32)
        
        if self.config.use_risk_metrics:
            # Rolling metrics based on the aggregated portfolio returns
            pf_returns = np.dot(self.returns.iloc[start_idx:end_idx], self.previous_weights)
            rm = [
                self.risk_computer.compute_rolling_volatility(pf_returns),
                self.risk_computer.compute_rolling_sharpe(pf_returns, self.config.risk_free_rate),
                self.risk_computer.compute_rolling_sortino(pf_returns, self.config.risk_free_rate),
                self.risk_computer.compute_rolling_var(pf_returns),
                self.risk_computer.compute_rolling_cvar(pf_returns),
                self.risk_computer.max_drawdown(np.array(self.portfolio_history)),
                self.risk_computer.calmar_ratio(np.array(self.portfolio_history), self.config.risk_free_rate),
            ]
            obs["risk_metrics"] = np.array(rm, dtype=np.float32)
        
        if self.config.use_time_freq and self.config.tech_indicator_config.use_time_freq:
            tf_matrix = []
            for asset in range(self.config.num_assets):
                tf = self.features[asset].get('time_freq', np.zeros(4, dtype=np.float32))
                tf_matrix.append(tf)
            obs["time_freq"] = np.array(tf_matrix, dtype=np.float32)
        
        return obs
    
    def render(self, mode='human'):
        plt.clear_figure()
        plt.plot(self.portfolio_history)
        plt.title("Live Portfolio Value")
        plt.show()
    
    def close(self):
        pass


def run_equal_weight_baseline(prices: pd.DataFrame, returns: pd.DataFrame, config: PortfolioConfig):
    """
    Run an equal-weight strategy as a baseline on the given data.
    Returns the final portfolio value and the portfolio history.
    """
    weights = np.ones(config.num_assets) / config.num_assets
    portfolio_value = config.initial_balance
    portfolio_history = [portfolio_value]
    
    for step in range(config.window_size, len(prices) - 1):
        current_returns = returns.iloc[step].values
        portfolio_return = np.dot(weights, current_returns)
        portfolio_value *= (1 + portfolio_return)
        portfolio_history.append(portfolio_value)
    
    total_return = portfolio_value / config.initial_balance - 1
    return total_return, portfolio_history
