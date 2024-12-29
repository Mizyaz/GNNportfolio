# feature_computer.py

import numpy as np
import talib
from typing import Dict, Optional
from env.config import TechnicalIndicatorConfig

class FeatureComputer:
    """Computes technical indicators and features using TA-Lib."""

    def __init__(self, config: TechnicalIndicatorConfig):
        self.config = config
    
    def compute_moving_averages(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        features = {}
        for period in self.config.sma_periods:
            features[f'sma_{period}'] = talib.SMA(prices, timeperiod=period)
        for period in self.config.ema_periods:
            features[f'ema_{period}'] = talib.EMA(prices, timeperiod=period)
        return features
    
    def compute_oscillators(self,
                            prices: np.ndarray,
                            high: Optional[np.ndarray] = None,
                            low: Optional[np.ndarray] = None,
                            volume: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        features = {}
        
        # RSI
        features['rsi'] = talib.RSI(prices, timeperiod=self.config.rsi_period)
        
        # MACD
        macd, signal, hist = talib.MACD(
            prices,
            fastperiod=self.config.macd_fast,
            slowperiod=self.config.macd_slow,
            signalperiod=self.config.macd_signal
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # MFI, CCI, Stoch (needs OHLCV)
        if high is not None and low is not None and volume is not None:
            features['mfi'] = talib.MFI(high, low, prices, volume, timeperiod=self.config.mfi_period)
            features['cci'] = talib.CCI(high, low, prices, timeperiod=self.config.cci_period)
            slowk, slowd = talib.STOCH(
                high, low, prices,
                fastk_period=self.config.stoch_k,
                slowk_period=self.config.stoch_slow,
                slowk_matype=0,
                slowd_period=self.config.stoch_d,
                slowd_matype=0
            )
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
        
        return features
    
    def compute_volatility(self,
                           prices: np.ndarray,
                           high: Optional[np.ndarray] = None,
                           low: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        features = {}
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            prices,
            timeperiod=self.config.bbands_period,
            nbdevup=self.config.bbands_dev,
            nbdevdn=self.config.bbands_dev,
            matype=0
        )
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        
        # ATR
        if high is not None and low is not None:
            features['atr'] = talib.ATR(high, low, prices, timeperiod=self.config.atr_period)
        
        return features
    
    def compute_momentum(self,
                         prices: np.ndarray,
                         high: Optional[np.ndarray] = None,
                         low: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        features = {}
        
        # ADX
        if high is not None and low is not None:
            features['adx'] = talib.ADX(high, low, prices, timeperiod=self.config.adx_period)
            
            # Aroon
            aroon_up, aroon_down = talib.AROON(high, low, timeperiod=self.config.aroon_period)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down
        
        return features
    
    def compute_time_frequency_features(self, prices: np.ndarray) -> np.ndarray:
        """Compute time-frequency features from a price series."""
        from scipy import signal
        
        if len(prices) < self.config.freq_window + 1:
            return np.zeros(4, dtype=np.float32)  # Return zeros if not enough data
        
        # Compute returns as log differences
        returns = np.diff(np.log(np.clip(prices, a_min=1e-7, a_max=None)))
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(returns,
                                       fs=1.0,
                                       nperseg=self.config.freq_window,
                                       noverlap=self.config.freq_overlap)
        
        if Sxx.size == 0:
            return np.zeros(4, dtype=np.float32)
        
        # Extract four summary features from the most recent time slice
        features = np.array([
            float(np.mean(Sxx, axis=0)[-1]),  # Latest mean
            float(np.std(Sxx, axis=0)[-1]),   # Latest std
            float(np.max(Sxx, axis=0)[-1]),   # Latest max
            float(f[np.argmax(Sxx[:, -1])])   # Latest dominant frequency
        ], dtype=np.float32)
        
        features = np.nan_to_num(features, 0.0)
        return features
