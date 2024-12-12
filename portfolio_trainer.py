import os
from typing import Dict, Any, Optional, List, NamedTuple, Tuple
from dataclasses import dataclass
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from financial_env import FinancialEnv, EnvironmentConfig, ObservationType, RewardType
from data_manager import DataManager

@dataclass
class DataConfig:
    """Configuration for data loading"""
    tickers: List[str] = None
    num_assets: int = 10
    train_start_date: str = "2020-01-01"
    train_end_date: str = "2022-12-31"
    val_start_date: str = "2022-01-01"
    val_end_date: str = "2023-12-31"
    force_reload: bool = False

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'PG', 'MA',
                'HD', 'BAC', 'CVX', 'KO', 'PFE', 'DIS', 'WMT', 'MRK', 'XOM', 'ORCL'
            ]

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    total_timesteps: int = 1_000_000
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    save_path: str = "./models"
    use_wandb: bool = True
    project_name: str = "portfolio_optimization"

@dataclass
class ModelConfig:
    """Configuration for PPO model parameters"""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01

class PortfolioCallback(BaseCallback):
    """Custom callback for logging portfolio metrics"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.returns = []
        self.sharpe_ratios = []
        self.batch_returns = []  # Returns for current batch
        self.batch_equal_weight = []  # Equal weight returns for current batch
        self.step_count = 0
        
    def _on_step(self) -> bool:
        """Called after each step"""
        info = self.locals['infos'][0]  # Get info from first environment
        env = self.training_env.envs[0].env
        
        # Calculate equal weight portfolio return
        num_assets = env.config.num_assets
        equal_weights = np.ones(num_assets) / num_assets
        
        # Get K-day returns from environment
        k_day_returns = info['k_day_returns']
        portfolio_return = np.mean(k_day_returns)
        
        # Calculate equal weight returns for the same period
        current_returns = env.returns_history[-1]
        equal_weight_return = np.dot(equal_weights, current_returns)
        
        # Store returns for batch comparison
        self.batch_returns.append(portfolio_return)
        self.batch_equal_weight.append(equal_weight_return)
        self.step_count += 1
        
        # Log metrics
        self.returns.append(portfolio_return)
        self.sharpe_ratios.append(info['reward_components'].get('sharpe', 0))
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                'portfolio_return': portfolio_return,
                'equal_weight_return': equal_weight_return,
                'return_difference': portfolio_return - equal_weight_return,
                'sharpe_ratio': info['reward_components'].get('sharpe', 0),
                'diversification': info['reward_components'].get('diversification', 0),
                'weights_std': np.std(info['weights']),
                'k_day_returns_mean': np.mean(k_day_returns),
                'k_day_returns_std': np.std(k_day_returns)
            })
        
        # Print batch statistics every n_steps
        if self.step_count % self.model.n_steps == 0:
            batch_portfolio_return = np.mean(self.batch_returns)
            batch_equal_weight_return = np.mean(self.batch_equal_weight)
            return_difference = batch_portfolio_return - batch_equal_weight_return
            
            # Calculate actual days covered
            days_per_step = env.config.prediction_days
            total_days = len(self.batch_returns) * days_per_step
            start_idx = env.current_idx - (total_days)
            end_idx = env.current_idx
            
            print(f"\nBatch Performance Summary:")
            print(f"Steps: {self.step_count - self.model.n_steps + 1}-{self.step_count}")
            print(f"Data Indices: {start_idx}-{end_idx} (Covering {total_days} trading days)")
            print(f"Average Daily Returns:")
            print(f"  Agent Portfolio: {batch_portfolio_return:.4f}")
            print(f"  Equal Weight: {batch_equal_weight_return:.4f}")
            print(f"  Difference: {return_difference:.4f} ({(return_difference/abs(batch_equal_weight_return))*100:.2f}%)")
            
            # Calculate cumulative returns for the batch period
            cumulative_portfolio = np.prod(1 + np.array(self.batch_returns)) - 1
            cumulative_equal = np.prod(1 + np.array(self.batch_equal_weight)) - 1
            cum_difference = cumulative_portfolio - cumulative_equal
            
            print(f"Cumulative Returns:")
            print(f"  Agent Portfolio: {cumulative_portfolio:.4f}")
            print(f"  Equal Weight: {cumulative_equal:.4f}")
            print(f"  Difference: {cum_difference:.4f} ({(cum_difference/abs(cumulative_equal))*100:.2f}%)")
            
            # Reset batch tracking
            self.batch_returns = []
            self.batch_equal_weight = []
        
        return True

class DataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_manager = DataManager()
    
    def load_data(self):
        """Load training and validation data"""
        # Load training data
        train_data = self._load_period_data(
            self.config.train_start_date,
            self.config.train_end_date
        )
        
        # Load validation data
        val_data = self._load_period_data(
            self.config.val_start_date,
            self.config.val_end_date
        )
        
        return train_data, val_data
    
    def _load_period_data(self, start_date: str, end_date: str):
        """Load data for a specific period"""
        X, y, selected_tickers = self.data_manager.load_data(
            tickers=self.config.tickers,
            num_assets=self.config.num_assets,
            start_date=start_date,
            end_date=end_date,
            force_reload=self.config.force_reload
        )
        return X, y, selected_tickers

class PortfolioTrainer:
    """Trainer for portfolio optimization using PPO"""
    
    def __init__(
        self,
        env_config: EnvironmentConfig,
        model_config: ModelConfig,
        train_data: tuple,
        val_data: Optional[tuple] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """Initialize trainer with configurations"""
        self.env_config = env_config
        self.model_config = model_config
        self.train_data = train_data
        self.val_data = val_data
        self.training_config = training_config or TrainingConfig()
        
        # Initialize environments
        self.train_env = self._create_env(train_data)
        self.eval_env = self._create_env(val_data) if val_data else None
        
        # Initialize model
        self.model = self._create_model()
        
    def _create_env(self, data: tuple) -> VecNormalize:
        """Create and wrap environment"""
        prices, returns, tickers = data
        
        def make_env():
            env = FinancialEnv(self.env_config, prices, returns)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.
        )
        return env
    
    def _create_model(self) -> PPO:
        """Create PPO model"""
        return PPO(
            "MultiInputPolicy",
            self.train_env,
            verbose=1,
            **vars(self.model_config)
        )
    
    def train(self):
        """Train the model"""
        # Initialize wandb
        if self.training_config.use_wandb:
            wandb.init(project=self.training_config.project_name)
        
        # Create callbacks
        callbacks = [PortfolioCallback()]
        
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=self.training_config.save_path,
                log_path=self.training_config.save_path,
                eval_freq=self.training_config.eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=self.training_config.n_eval_episodes
            )
            callbacks.append(eval_callback)
        
        # Train model
        self.model.learn(
            total_timesteps=self.training_config.total_timesteps,
            callback=callbacks
        )
        
        # Save final model
        if self.training_config.save_path:
            os.makedirs(self.training_config.save_path, exist_ok=True)
            self.model.save(os.path.join(self.training_config.save_path, "final_model"))
            self.train_env.save(os.path.join(self.training_config.save_path, "vec_normalize.pkl"))
        
        if self.training_config.use_wandb:
            wandb.finish()
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """Evaluate the model"""
        episode_rewards = []
        episode_lengths = []
        portfolio_returns = []
        sharpe_ratios = []
        
        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_returns = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_returns.append(info[0]['portfolio_return'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_returns.append(np.mean(episode_returns))
            sharpe_ratios.append(
                np.mean(episode_returns) / (np.std(episode_returns) + 1e-6)
            )
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_return': np.mean(portfolio_returns),
            'sharpe_ratio': np.mean(sharpe_ratios)
        }

def main():
    """Example usage with robust error handling"""
    try:
        # Create configurations
        data_config = DataConfig()
        
        env_config = EnvironmentConfig(
            window_size=20,
            prediction_days=5,
            num_assets=data_config.num_assets,
            observation_types=[ObservationType.PRICES, ObservationType.RETURNS],
            reward_types=[RewardType.SHARPE, RewardType.DIVERSIFICATION],
            transaction_cost=0.0,
            risk_free_rate=0.0,
            target_volatility=0.0,
            regularization_factor=0.0
        )
        
        model_config = ModelConfig()
        training_config = TrainingConfig()
        
        # Load data
        data_loader = DataLoader(data_config)
        train_data, val_data = data_loader.load_data()
        
        # Create and train model
        trainer = PortfolioTrainer(
            env_config=env_config,
            model_config=model_config,
            train_data=train_data,
            val_data=val_data,
            training_config=training_config
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        results = trainer.evaluate(n_episodes=10)
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()