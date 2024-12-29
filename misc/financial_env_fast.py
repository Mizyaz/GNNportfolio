import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import talib
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotext as plt

@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicators"""
    # Moving Averages
    sma_periods: List[int] = (20, 50, 200)
    ema_periods: List[int] = (12, 26)
    
    # Oscillators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volatility
    bbands_period: int = 20
    bbands_dev: float = 2.0
    atr_period: int = 14
    
    # Volume & Momentum
    obv_enabled: bool = True
    adx_period: int = 14
    aroon_period: int = 14
    cci_period: int = 14
    mfi_period: int = 14
    
    # Additional
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_slow: int = 3
    
    # Time Frequency
    use_time_freq: bool = True
    freq_window: int = 20
    freq_overlap: int = 10

@dataclass
class PortfolioConfig:
    """Configuration for portfolio parameters"""
    window_size: int = 50
    num_assets: int = 10
    initial_balance: float = 100000
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001
    use_technical_indicators: bool = True
    use_correlation_features: bool = True
    use_risk_metrics: bool = True
    use_time_freq: bool = True
    tech_indicator_config: TechnicalIndicatorConfig = TechnicalIndicatorConfig()

class FeatureComputer:
    """Computes technical indicators and features using TA-Lib"""
    
    def __init__(self, config: TechnicalIndicatorConfig):
        self.config = config
    
    def compute_moving_averages(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        features = {}
        for period in self.config.sma_periods:
            features[f'sma_{period}'] = talib.SMA(prices, timeperiod=period)
        for period in self.config.ema_periods:
            features[f'ema_{period}'] = talib.EMA(prices, timeperiod=period)
        return features
    
    def compute_oscillators(self, prices: np.ndarray, high: Optional[np.ndarray] = None, 
                          low: Optional[np.ndarray] = None, volume: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        features = {}
        
        # RSI
        features['rsi'] = talib.RSI(prices, timeperiod=self.config.rsi_period)
        
        # MACD
        macd, signal, hist = talib.MACD(prices, 
                                      fastperiod=self.config.macd_fast,
                                      slowperiod=self.config.macd_slow,
                                      signalperiod=self.config.macd_signal)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        if all(x is not None for x in [high, low, volume]):
            # MFI
            features['mfi'] = talib.MFI(high, low, prices, volume, timeperiod=self.config.mfi_period)
            
            # CCI
            features['cci'] = talib.CCI(high, low, prices, timeperiod=self.config.cci_period)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, prices,
                                     fastk_period=self.config.stoch_k,
                                     slowk_period=self.config.stoch_slow,
                                     slowk_matype=0,
                                     slowd_period=self.config.stoch_d,
                                     slowd_matype=0)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
        
        return features
    
    def compute_volatility(self, prices: np.ndarray, high: Optional[np.ndarray] = None,
                         low: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        features = {}
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(prices,
                                          timeperiod=self.config.bbands_period,
                                          nbdevup=self.config.bbands_dev,
                                          nbdevdn=self.config.bbands_dev,
                                          matype=0)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        
        if high is not None and low is not None:
            # ATR
            features['atr'] = talib.ATR(high, low, prices, timeperiod=self.config.atr_period)
            
        return features
    
    def compute_momentum(self, prices: np.ndarray, high: Optional[np.ndarray] = None,
                       low: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        features = {}
        
        if high is not None and low is not None:
            # ADX
            features['adx'] = talib.ADX(high, low, prices, timeperiod=self.config.adx_period)
            
            # Aroon
            aroon_up, aroon_down = talib.AROON(high, low, timeperiod=self.config.aroon_period)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down
            
        return features
    
    def compute_time_frequency_features(self, prices: np.ndarray) -> np.ndarray:
        """Compute time frequency features"""
        try:
            from scipy import signal
            
            # Ensure we have enough data
            if len(prices) < self.config.freq_window + 1:
                return np.zeros(4, dtype=np.float32)  # Return zeros if not enough data
            
            # Compute returns safely
            returns = np.diff(np.log(np.clip(prices, a_min=1e-7, a_max=None)))
            
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, 
                                         nperseg=self.config.freq_window,
                                         noverlap=self.config.freq_overlap)
            
            if Sxx.size == 0:  # Check if spectrogram is empty
                return np.zeros(4, dtype=np.float32)
            
            # Extract features safely
            features = np.array([
                float(np.mean(Sxx, axis=0)[-1]),  # Latest mean
                float(np.std(Sxx, axis=0)[-1]),   # Latest std
                float(np.max(Sxx, axis=0)[-1]),   # Latest max
                float(f[np.argmax(Sxx[:, -1])])   # Latest dominant frequency
            ], dtype=np.float32)
            
            # Replace any invalid values with 0
            features = np.nan_to_num(features, 0.0)
            
            return features
            
        except Exception as e:
            print(f"Warning: Error computing time frequency features: {str(e)}")
            return np.zeros(4, dtype=np.float32)

class RiskMetricsComputer:
    """Computes various risk metrics for the portfolio"""
    
    @staticmethod
    def compute_rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        return np.std(returns[-window:]) * np.sqrt(252)
    
    @staticmethod
    def compute_rolling_sharpe(returns: np.ndarray, risk_free_rate: float, window: int = 20) -> np.ndarray:
        excess_returns = returns - risk_free_rate/252
        return np.mean(excess_returns[-window:]) / (np.std(excess_returns[-window:]) + 1e-8) * np.sqrt(252)
    
    @staticmethod
    def compute_rolling_sortino(returns: np.ndarray, risk_free_rate: float, window: int = 20) -> np.ndarray:
        excess_returns = returns[-window:] - risk_free_rate/252
        downside_returns = np.where(excess_returns < 0, excess_returns, 0)
        downside_std = np.std(downside_returns) + 1e-8
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    @staticmethod
    def compute_rolling_var(returns: np.ndarray, window: int = 20, confidence: float = 0.95) -> np.ndarray:
        return -np.percentile(returns[-window:], (1 - confidence) * 100)
    
    @staticmethod
    def compute_rolling_cvar(returns: np.ndarray, window: int = 20, confidence: float = 0.95) -> np.ndarray:
        var = -np.percentile(returns[-window:], (1 - confidence) * 100)
        return -np.mean(returns[-window:][returns[-window:] <= -var])

class PortfolioEnvFast(gym.Env):
    """Enhanced Portfolio Environment with Dictionary Observation Space"""
    
    def __init__(self, config: PortfolioConfig,
                 prices: pd.DataFrame,
                 returns: pd.DataFrame,
                 high_prices: Optional[pd.DataFrame] = None,
                 low_prices: Optional[pd.DataFrame] = None,
                 volumes: Optional[pd.DataFrame] = None):
        """Initialize the environment with proper error handling"""
        super().__init__()
        
        # Initialize configs and computers first
        self.config = config
        self.feature_computer = FeatureComputer(config.tech_indicator_config)
        self.risk_computer = RiskMetricsComputer()
        
        # Initialize data attributes with validation
        self.prices = prices.copy()
        self.returns = returns.copy()
        self.high_prices = high_prices.copy() if high_prices is not None else None
        self.low_prices = low_prices.copy() if low_prices is not None else None
        self.volumes = volumes.copy() if volumes is not None else None
        
        # Validate data
        if len(prices.columns) != config.num_assets:
            raise ValueError(f"Number of assets in prices ({len(prices.columns)}) does not match config ({config.num_assets})")
        
        # Initialize state variables with defaults
        self.current_step = self.config.window_size
        self.portfolio_value = self.config.initial_balance
        self.previous_weights = np.array([1.0 / self.config.num_assets] * self.config.num_assets)
        self.portfolio_history = [self.portfolio_value]
        
        # Initialize features dictionary
        self.features = {asset: {} for asset in range(self.config.num_assets)}
        
        # Setup observation and action spaces
        self._setup_observation_space()
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(self.config.num_assets,),
            dtype=np.float32
        )

        self.episode_count = 0
        self.portfolio_history_global = []
        self.all_portfolio_histories = []
        
        # Precompute features after all initializations
        if self.config.use_technical_indicators:
            self._precompute_features()
    
    def _get_observation_keys(self) -> Dict[str, Tuple[int]]:
        """Get all possible observation keys and their shapes"""
        # Calculate number of technical indicators
        n_basic = 0
        if self.config.use_technical_indicators:
            n_basic = (len(self.config.tech_indicator_config.sma_periods) + 
                      len(self.config.tech_indicator_config.ema_periods) + 
                      7)  # RSI, MACD (3 values), BB (3 values)
            
            if self.high_prices is not None and self.low_prices is not None:
                n_basic += 6  # ATR, ADX, Aroon (2 values), CCI, MFI
            
            if self.volumes is not None:
                n_basic += 2  # OBV, Volume SMA
        
        obs_keys = {
            'prices': (self.config.window_size, self.config.num_assets),
            'returns': (self.config.window_size, self.config.num_assets),
            'weights': (self.config.num_assets,),
            'portfolio_value': (1,)
        }
        
        if self.config.use_technical_indicators and n_basic > 0:
            obs_keys['technical_indicators'] = (self.config.num_assets, n_basic)
        
        if self.config.use_correlation_features:
            n_corr = self.config.num_assets * (self.config.num_assets - 1) // 2
            obs_keys['correlation'] = (n_corr,)
        
        if self.config.use_risk_metrics:
            obs_keys['risk_metrics'] = (5,)  # Volatility, Sharpe, Sortino, VaR, CVaR
        
        if self.config.use_time_freq and self.config.tech_indicator_config.use_time_freq:
            obs_keys['time_freq'] = (self.config.num_assets, 4)
        
        return obs_keys
    
    def _setup_observation_space(self):
        """Setup the observation space as a Dict space"""
        obs_keys = self._get_observation_keys()
        
        # Create the observation space dictionary
        spaces_dict = {}
        for key, shape in obs_keys.items():
            spaces_dict[key] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32
            )
        
        self.observation_space = spaces.Dict(spaces_dict)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation state as a dictionary"""
        try:
            start_idx = self.current_step - self.config.window_size
            end_idx = self.current_step
            
            # Initialize observation dictionary with basic features
            obs = {
                'prices': np.nan_to_num(self.prices.iloc[start_idx:end_idx].values, 0.0),
                'returns': np.nan_to_num(self.returns.iloc[start_idx:end_idx].values, 0.0),
                'weights': self.previous_weights,
                'portfolio_value': np.array([self.portfolio_value], dtype=np.float32)
            }
            
            if self.config.use_technical_indicators:
                tech_indicators = []
                for asset in range(self.config.num_assets):
                    asset_features = self.features.get(asset, {})
                    asset_indicators = []
                    
                    # Add all technical indicators in a fixed order
                    # SMA
                    for period in self.config.tech_indicator_config.sma_periods:
                        key = f'sma_{period}'
                        if key in asset_features:
                            value = asset_features[key][self.current_step] if self.current_step < len(asset_features[key]) else 0.0
                            asset_indicators.append(float(np.nan_to_num(value, 0.0)))
                        else:
                            asset_indicators.append(0.0)
                    
                    # EMA
                    for period in self.config.tech_indicator_config.ema_periods:
                        key = f'ema_{period}'
                        if key in asset_features:
                            value = asset_features[key][self.current_step] if self.current_step < len(asset_features[key]) else 0.0
                            asset_indicators.append(float(np.nan_to_num(value, 0.0)))
                        else:
                            asset_indicators.append(0.0)
                    
                    # Other basic indicators
                    basic_indicators = ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                                     'bb_upper', 'bb_middle', 'bb_lower']
                    for ind in basic_indicators:
                        if ind in asset_features:
                            value = asset_features[ind][self.current_step] if self.current_step < len(asset_features[ind]) else 0.0
                            asset_indicators.append(float(np.nan_to_num(value, 0.0)))
                        else:
                            asset_indicators.append(0.0)
                    
                    # OHLCV indicators if available
                    if self.high_prices is not None and self.low_prices is not None:
                        ohlcv_indicators = ['atr', 'adx', 'aroon_up', 'aroon_down', 'cci', 'mfi']
                        for ind in ohlcv_indicators:
                            if ind in asset_features:
                                value = asset_features[ind][self.current_step] if self.current_step < len(asset_features[ind]) else 0.0
                                asset_indicators.append(float(np.nan_to_num(value, 0.0)))
                            else:
                                asset_indicators.append(0.0)
                    
                    if self.volumes is not None:
                        volume_indicators = ['obv', 'volume_sma']
                        for ind in volume_indicators:
                            if ind in asset_features:
                                value = asset_features[ind][self.current_step] if self.current_step < len(asset_features[ind]) else 0.0
                                asset_indicators.append(float(np.nan_to_num(value, 0.0)))
                            else:
                                asset_indicators.append(0.0)
                    
                    tech_indicators.append(asset_indicators)
                
                if tech_indicators and tech_indicators[0]:  # Only add if we have indicators
                    obs['technical_indicators'] = np.array(tech_indicators, dtype=np.float32)
            
            if self.config.use_correlation_features:
                try:
                    corr_matrix = self.returns.iloc[start_idx:end_idx].corr().values
                    corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                    obs['correlation'] = np.nan_to_num(corr_values, 0.0)
                except Exception as e:
                    print(f"Warning: Error computing correlation features: {str(e)}")
                    n_corr = self.config.num_assets * (self.config.num_assets - 1) // 2
                    obs['correlation'] = np.zeros(n_corr, dtype=np.float32)
            
            if self.config.use_risk_metrics:
                try:
                    portfolio_returns = np.dot(self.returns.iloc[start_idx:end_idx], self.previous_weights)
                    risk_metrics = [
                        self.risk_computer.compute_rolling_volatility(portfolio_returns),
                        self.risk_computer.compute_rolling_sharpe(portfolio_returns, self.config.risk_free_rate),
                        self.risk_computer.compute_rolling_sortino(portfolio_returns, self.config.risk_free_rate),
                        self.risk_computer.compute_rolling_var(portfolio_returns),
                        self.risk_computer.compute_rolling_cvar(portfolio_returns)
                    ]
                    obs['risk_metrics'] = np.nan_to_num(risk_metrics, 0.0)
                except Exception as e:
                    print(f"Warning: Error computing risk metrics: {str(e)}")
                    obs['risk_metrics'] = np.zeros(5, dtype=np.float32)
            
            if self.config.use_time_freq and self.config.tech_indicator_config.use_time_freq:
                time_freq_features = []
                for asset in range(self.config.num_assets):
                    try:
                        if 'time_freq' in self.features[asset]:
                            time_freq = self.features[asset]['time_freq']
                            time_freq_features.append(time_freq)
                        else:
                            time_freq_features.append(np.zeros(4, dtype=np.float32))
                    except Exception as e:
                        print(f"Warning: Error getting time frequency features for asset {asset}: {str(e)}")
                        time_freq_features.append(np.zeros(4, dtype=np.float32))
                obs['time_freq'] = np.array(time_freq_features, dtype=np.float32)
            
            return obs
            
        except Exception as e:
            print(f"Warning: Error in _get_observation: {str(e)}")
            # Return zero-filled observation with correct shapes
            return {key: np.zeros(shape, dtype=np.float32) 
                   for key, shape in self._get_observation_keys().items()}
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = self.config.window_size
        self.portfolio_value = self.config.initial_balance
        self.previous_weights = np.array([1.0 / self.config.num_assets] * self.config.num_assets)
        self.portfolio_history = [self.portfolio_value]
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        # Normalize weights
        weights = action / (np.sum(action) + 1e-8)
        
        # Calculate transaction costs
        turnover = np.sum(np.abs(weights - self.previous_weights))
        transaction_costs = turnover * self.config.transaction_cost
        
        # Get returns and update portfolio value
        current_returns = self.returns.iloc[self.current_step].values
        portfolio_return = np.dot(weights, current_returns) - transaction_costs
        self.portfolio_value *= (1 + portfolio_return)
        
        # Store history
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate reward (Sharpe Ratio)
        observation = self._get_observation()
        rolling_returns = self.returns.iloc[self.current_step-20:self.current_step]
        portfolio_returns = np.dot(rolling_returns, weights)
        reward = self.risk_computer.compute_rolling_sharpe(portfolio_returns, self.config.risk_free_rate)
        
        # Update state
        self.previous_weights = weights
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.prices) - 1

        if done:
            print(f"Episode {self.episode_count} completed. Portfolio value: {self.portfolio_value:.2f}")
            self.episode_count += 1
            self.portfolio_history_global.append(self.portfolio_value)
            self.all_portfolio_histories.append(self.portfolio_history)
            plt.plot(self.portfolio_history)
            plt.title(f'Portfolio Value Over Time for episode {self.episode_count}')
            plt.xlabel('Steps')
            plt.ylabel('Portfolio Value ($)')
            plt.show()
            plt.clear_figure()

            if self.episode_count % 10 == 0:
                
                plt.plot(self.portfolio_history_global, color="red")
                plt.title(f'Portfolio Value Over Time for episode {self.episode_count} avg portfolio value {np.mean(self.portfolio_history_global):.2f}')
                plt.xlabel('Steps')
                plt.ylabel('Portfolio Value ($)')
                plt.show()
                plt.clear_figure()
                




        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_costs': transaction_costs,
            'weights': weights
        }
        
        return observation, reward, done, False, info
    
    def render(self, mode='human'):
        """Render the environment"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio_history)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.show()
    
    def close(self):
        """Close the environment"""
        plt.close()
    
    def _precompute_features(self):
        """Precompute technical indicators for faster access"""
        try:
            for asset in range(self.config.num_assets):
                asset_features = {}
                
                # Get asset data
                prices = self.prices.iloc[:, asset].values
                
                if self.config.use_technical_indicators:
                    # Compute basic indicators
                    try:
                        asset_features.update(self.feature_computer.compute_moving_averages(prices))
                    except Exception as e:
                        print(f"Warning: Error computing moving averages for asset {asset}: {str(e)}")
                    
                    try:
                        asset_features.update(self.feature_computer.compute_oscillators(prices))
                    except Exception as e:
                        print(f"Warning: Error computing oscillators for asset {asset}: {str(e)}")
                    
                    try:
                        asset_features.update(self.feature_computer.compute_volatility(prices))
                    except Exception as e:
                        print(f"Warning: Error computing volatility for asset {asset}: {str(e)}")
                    
                    # Compute OHLCV indicators if data available
                    if self.high_prices is not None and self.low_prices is not None:
                        high = self.high_prices.iloc[:, asset].values
                        low = self.low_prices.iloc[:, asset].values
                        volume = self.volumes.iloc[:, asset].values if self.volumes is not None else None
                        
                        try:
                            asset_features.update(self.feature_computer.compute_oscillators(prices, high, low, volume))
                        except Exception as e:
                            print(f"Warning: Error computing OHLCV oscillators for asset {asset}: {str(e)}")
                        
                        try:
                            asset_features.update(self.feature_computer.compute_volatility(prices, high, low))
                        except Exception as e:
                            print(f"Warning: Error computing OHLCV volatility for asset {asset}: {str(e)}")
                        
                        try:
                            asset_features.update(self.feature_computer.compute_momentum(prices, high, low))
                        except Exception as e:
                            print(f"Warning: Error computing momentum for asset {asset}: {str(e)}")
                
                # Compute time frequency features if enabled
                if self.config.use_time_freq and self.config.tech_indicator_config.use_time_freq:
                    try:
                        asset_features['time_freq'] = self.feature_computer.compute_time_frequency_features(prices)
                    except Exception as e:
                        print(f"Warning: Error computing time frequency features for asset {asset}: {str(e)}")
                        asset_features['time_freq'] = np.zeros(4, dtype=np.float32)
                
                self.features[asset] = asset_features
                
        except Exception as e:
            print(f"Warning: Error in precomputing features: {str(e)}")
            # Ensure features dictionary is initialized even if computation fails
            self.features = {asset: {} for asset in range(self.config.num_assets)} 