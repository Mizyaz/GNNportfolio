# config.py

from dataclasses import dataclass
from typing import List
from env.reward import RewardConfig
@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicators."""
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
    """Configuration for portfolio parameters."""
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
    reward_config: RewardConfig = RewardConfig(),
    use_per_episode_plot: bool = True