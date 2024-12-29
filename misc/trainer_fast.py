import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from financial_env_fast import PortfolioEnvFast, PortfolioConfig, TechnicalIndicatorConfig
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
import pickle 
import time
import os
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from gymnasium import spaces

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    total_timesteps: int = 100000
    eval_episodes: int = 5
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    save_freq: int = 10000
    log_interval: int = 100
    n_envs: int = 4  # Number of parallel environments
    deterministic_eval: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

@dataclass
class ModelConfig:
    """Configuration for the RL model"""
    algorithm: str = "PPO"  # One of "PPO", "A2C", "SAC"
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    policy_kwargs: Dict = None
    verbose: int = 1

class TrainingCallback(gym.Wrapper):
    """Enhanced callback for logging metrics during training"""
    def __init__(self, env, log_dir: str, verbose: int = 1):
        super(TrainingCallback, self).__init__(env)
        self.verbose = verbose
        self.episode_returns = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.transaction_costs = []
        self.sharpe_ratios = []
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "training.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def reset(self, **kwargs):
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.current_transaction_costs = 0.0
        return super().reset(**kwargs)
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        self.current_episode_return += reward
        self.current_episode_length += 1
        self.current_transaction_costs += info.get('transaction_costs', 0.0)
        
        if done:
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.current_episode_length)
            self.portfolio_values.append(self.env.portfolio_value)
            self.transaction_costs.append(self.current_transaction_costs)
            
            if self.verbose > 0 and len(self.episode_returns) % 10 == 0:
                avg_return = np.mean(self.episode_returns[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                portfolio_return = (self.env.portfolio_value / self.env.config.initial_balance - 1)
                
                msg = (f"Episode {len(self.episode_returns)} | "
                      f"Avg Return: {avg_return:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Portfolio Return: {portfolio_return:.2%}")
                print(msg)
                logging.info(msg)
        
        return observation, reward, done, truncated, info

class PortfolioTrainerFast:
    """Enhanced trainer class for the PortfolioEnvFast environment"""
    
    def __init__(self, 
                 portfolio_config: PortfolioConfig,
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 experiment_name: str = None):
        """
        Initialize the trainer.
        
        Args:
            portfolio_config: Configuration for the portfolio environment
            model_config: Configuration for the RL model
            training_config: Configuration for the training process
            experiment_name: Name for the experiment (for logging)
        """
        self.portfolio_config = portfolio_config
        self.model_config = model_config
        self.training_config = training_config
        
        # Setup experiment name and directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        self.base_dir = Path(f"results/{self.experiment_name}")
        self.model_dir = self.base_dir / "models"
        self.log_dir = self.base_dir / "logs"
        
        for dir_path in [self.base_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.env = None
        self.eval_env = None
    
    def _make_env(self, prices: pd.DataFrame, returns: pd.DataFrame,
                 high_prices: Optional[pd.DataFrame] = None,
                 low_prices: Optional[pd.DataFrame] = None,
                 volumes: Optional[pd.DataFrame] = None,
                 is_eval: bool = False) -> gym.Env:
        """Create and configure the environment"""
        env = PortfolioEnvFast(
            config=self.portfolio_config,
            prices=prices,
            returns=returns,
            high_prices=high_prices,
            low_prices=low_prices,
            volumes=volumes
        )
        
        # Convert Path to string for Monitor
        log_path = str(self.log_dir / ("eval" if is_eval else "train") / "monitor.csv")
        env = Monitor(env, log_path)
        return env
    
    def _create_vec_env(self, prices: pd.DataFrame, returns: pd.DataFrame,
                       high_prices: Optional[pd.DataFrame] = None,
                       low_prices: Optional[pd.DataFrame] = None,
                       volumes: Optional[pd.DataFrame] = None,
                       is_eval: bool = False) -> Union[DummyVecEnv, SubprocVecEnv]:
        """Create vectorized environment"""
        def make_env():
            return lambda: self._make_env(prices, returns, high_prices, low_prices, volumes, is_eval)
        
        # Always use DummyVecEnv for evaluation to avoid potential issues
        if is_eval:
            return DummyVecEnv([make_env()])
        
        # For training, use SubprocVecEnv if n_envs > 1, otherwise DummyVecEnv
        if self.training_config.n_envs > 1:
            return SubprocVecEnv([make_env() for _ in range(self.training_config.n_envs)])
        return DummyVecEnv([make_env()])
    
    def _create_model(self, env: gym.Env) -> Union[PPO, A2C, SAC]:
        """Create and configure the RL model with proper network architecture"""
        algorithms = {
            "PPO": PPO,
            "A2C": A2C,
            "SAC": SAC
        }
        
        Algorithm = algorithms.get(self.model_config.algorithm)
        if Algorithm is None:
            raise ValueError(f"Unknown algorithm: {self.model_config.algorithm}")
        
        # Configure policy network for dictionary observations
        policy_kwargs = self.model_config.policy_kwargs or {}
        
        # Get the observation space structure
        if isinstance(env.observation_space, spaces.Dict):
            # Calculate total features for each observation type
            feature_dims = {}
            for key, space in env.observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    feature_dims[key] = int(np.prod(space.shape))
            
            # Configure network architecture
            policy_kwargs.update({
                "net_arch": {
                    "pi": [256, 128, 64],  # Policy network
                    "vf": [256, 128, 64]   # Value network
                }
            })
        
        return Algorithm(
            policy="MultiInputPolicy",  # Use MultiInputPolicy for dictionary observations
            env=env,
            learning_rate=self.model_config.learning_rate,
            n_steps=self.model_config.n_steps,
            batch_size=self.model_config.batch_size,
            n_epochs=self.model_config.n_epochs,
            gamma=self.model_config.gamma,
            gae_lambda=self.model_config.gae_lambda,
            clip_range=self.model_config.clip_range,
            ent_coef=self.model_config.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=self.model_config.verbose,
            tensorboard_log=str(self.log_dir)
        )
    
    @staticmethod
    def compute_time_frequency_features(prices: pd.DataFrame, window_size: int = 20) -> Dict[str, np.ndarray]:
        """Compute time frequency features for each asset"""
        from scipy import signal
        
        features = []
        for col in prices.columns:
            # Get the price series for the asset
            price_series = prices[col].values
            
            # Compute returns
            returns = np.diff(np.log(price_series))
            
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, nperseg=window_size, noverlap=window_size//2)
            
            # Extract features from spectrogram
            features.extend([
                np.mean(Sxx, axis=0)[-1],  # Latest mean
                np.std(Sxx, axis=0)[-1],   # Latest std
                np.max(Sxx, axis=0)[-1],   # Latest max
                f[np.argmax(Sxx[:, -1])]   # Latest dominant frequency
            ])
            
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def download_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download and preprocess data from yfinance with caching"""
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Create cache filename based on parameters
        cache_file = data_dir / f"market_data_{'-'.join(sorted(tickers))}_{start_date}_{end_date}.pkl"
        
        # Check if cached data exists
        if cache_file.exists():
            print("Loading data from cache...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Verify data structure
            expected_columns = ['Close', 'High', 'Low', 'Volume']
            if all(col in cached_data.columns.levels[0] for col in expected_columns):
                print("Cache validation successful.")
                data = cached_data
            else:
                print("Cache validation failed. Downloading fresh data...")
                data = yf.download(tickers, start=start_date, end=end_date)
        else:
            print(f"Downloading data for {len(tickers)} assets...")
            data = yf.download(tickers, start=start_date, end=end_date)
            
            # Cache the raw data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print("Data cached for future use.")
        
        # Extract and process data
        prices = data['Close'].astype(np.float64)
        high_prices = data['High'].astype(np.float64)
        low_prices = data['Low'].astype(np.float64)
        volumes = data['Volume'].astype(np.float64)
        
        # Calculate returns using close prices
        returns = prices.pct_change()
        
        # Handle missing data
        for df in [prices, returns, high_prices, low_prices, volumes]:
            df.ffill(inplace=True)
            df.fillna(0, inplace=True)
        
        print("Data preprocessing complete.")
        return prices, returns, high_prices, low_prices, volumes
    
    def train(self, train_data: Tuple[pd.DataFrame, ...], eval_data: Optional[Tuple[pd.DataFrame, ...]] = None):
        """Train the model with proper error handling"""
        try:
            # Unpack data
            train_prices, train_returns = train_data[0], train_data[1]
            train_high, train_low, train_volumes = (None, None, None) if len(train_data) <= 2 else train_data[2:]
            
            # Create environments
            self.env = self._create_vec_env(train_prices, train_returns, train_high, train_low, train_volumes)
            
            callbacks = []
            if eval_data is not None:
                eval_prices, eval_returns = eval_data[0], eval_data[1]
                eval_high, eval_low, eval_volumes = (None, None, None) if len(eval_data) <= 2 else eval_data[2:]
                
                self.eval_env = self._create_vec_env(eval_prices, eval_returns, eval_high, eval_low, eval_volumes, is_eval=True)
                
                # Setup callbacks
                eval_callback = EvalCallback(
                    self.eval_env,
                    best_model_save_path=str(self.model_dir),
                    log_path=str(self.log_dir),
                    eval_freq=self.training_config.eval_freq,
                    deterministic=self.training_config.deterministic_eval,
                    render=False,
                    n_eval_episodes=self.training_config.n_eval_episodes
                )
                callbacks.append(eval_callback)
            
            # Create and train model
            self.model = self._create_model(self.env)
            
            print(f"Starting training for {self.training_config.total_timesteps} timesteps...")
            self.model.learn(
                total_timesteps=self.training_config.total_timesteps,
                callback=callbacks,
                log_interval=self.training_config.log_interval,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = self.model_dir / "final_model.zip"
            self.model.save(str(final_model_path))
            print(f"Final model saved to {final_model_path}")
            
            # Save configurations
            self._save_configs()
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise
    
    def _save_configs(self):
        """Save all configurations"""
        configs = {
            'portfolio_config': self.portfolio_config,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        config_path = self.base_dir / "configs.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(configs, f)
    
    def evaluate(self, eval_data: Tuple[pd.DataFrame, ...], n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # Unpack evaluation data
        eval_prices, eval_returns = eval_data[0], eval_data[1]
        eval_high, eval_low, eval_volumes = (None, None, None) if len(eval_data) <= 2 else eval_data[2:]
        
        # Create evaluation environment
        eval_env = self._make_env(eval_prices, eval_returns, eval_high, eval_low, eval_volumes, is_eval=True)
        
        results = {
            'returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'portfolio_values': []
        }
        
        print(f"\nEvaluating model over {n_episodes} episodes...")
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
                episode_reward += reward
            
            portfolio_return = eval_env.unwrapped.portfolio_value / eval_env.unwrapped.config.initial_balance - 1
            results['returns'].append(portfolio_return)
            results['portfolio_values'].append(eval_env.portfolio_history)
            
            print(f"Episode {episode + 1}: Return = {portfolio_return:.2%}")
        
        # Compute aggregate statistics
        mean_return = np.mean(results['returns'])
        std_return = np.std(results['returns'])
        
        print(f"\nEvaluation Results:")
        print(f"Mean Return: {mean_return:.2%}")
        print(f"Std Return: {std_return:.2%}")
        
        # Plot results
        self._plot_evaluation_results(results)
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'raw_results': results
        }
    
    def _plot_evaluation_results(self, results: Dict[str, List]):
        """Plot evaluation results"""
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio values for each episode
        for i, values in enumerate(results['portfolio_values']):
            plt.plot(values, alpha=0.5, label=f'Episode {i+1}')
        
        plt.title('Portfolio Value Over Time (All Episodes)')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(self.log_dir / 'evaluation_results.png')
        plt.close()

def run_experiment(
    tickers: List[str],
    train_dates: Tuple[str, str],
    test_dates: Tuple[str, str],
    portfolio_config: Optional[PortfolioConfig] = None,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    experiment_name: Optional[str] = None
) -> Dict[str, float]:
    """Run a complete experiment"""
    
    # Use default configs if not provided
    portfolio_config = portfolio_config or PortfolioConfig(num_assets=len(tickers))
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    
    # Initialize trainer
    trainer = PortfolioTrainerFast(
        portfolio_config=portfolio_config,
        model_config=model_config,
        training_config=training_config,
        experiment_name=experiment_name
    )
    
    # Download data
    train_data = trainer.download_data(tickers, *train_dates)
    test_data = trainer.download_data(tickers, *test_dates)
    
    # Train model
    trainer.train(train_data, test_data)
    
    # Evaluate model
    results = trainer.evaluate(test_data, n_episodes=training_config.eval_episodes)
    
    return results

if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'NVDA', 'TSLA', 'JPM', 'JNJ', 'PG']
        
    portfolio_config = PortfolioConfig(
        window_size=20,
        num_assets=len(tickers),
        initial_balance=100,
        risk_free_rate=0.02,
        transaction_cost=0.001,
        use_technical_indicators=True,
        use_correlation_features=False,
        use_risk_metrics=False,
        tech_indicator_config=TechnicalIndicatorConfig(use_time_freq=False)
    )
    
    model_config = ModelConfig(
        algorithm="PPO",
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0
    )
    
    training_config = TrainingConfig(
        total_timesteps=100000,
        eval_episodes=5,
        n_envs=1,
    )
    
    results = run_experiment(
        tickers=tickers,
        train_dates=("2017-01-01", "2018-01-01"),
        test_dates=("2016-01-01", "2017-01-01"),
        portfolio_config=portfolio_config,
        model_config=model_config,
        training_config=training_config,
        experiment_name="enhanced_portfolio_optimization"
    ) 