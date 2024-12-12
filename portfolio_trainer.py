import os
from typing import Dict, Any, Optional, List
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from financial_env import FinancialEnv, EnvironmentConfig
from data_manager import DataManager

class PortfolioCallback(BaseCallback):
    """Custom callback for logging portfolio metrics"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.returns = []
        self.sharpe_ratios = []
        self.turnovers = []
        
    def _on_step(self) -> bool:
        """Called after each step"""
        info = self.locals['infos'][0]  # Get info from first environment
        
        # Log metrics
        self.returns.append(info['portfolio_return'])
        self.sharpe_ratios.append(info['reward_components'].get('sharpe', 0))
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                'portfolio_return': info['portfolio_return'],
                'sharpe_ratio': info['reward_components'].get('sharpe', 0),
                'diversification': info['reward_components'].get('diversification', 0),
                'weights_std': np.std(info['weights'])
            })
        
        return True

class PortfolioTrainer:
    """Trainer for portfolio optimization using PPO"""
    
    def __init__(
        self,
        config: EnvironmentConfig,
        model_params: Dict[str, Any],
        train_data: tuple,
        val_data: Optional[tuple] = None,
        use_wandb: bool = False,
        project_name: str = "portfolio_optimization"
    ):
        """
        Initialize trainer
        
        Parameters:
        -----------
        config : EnvironmentConfig
            Environment configuration
        model_params : Dict[str, Any]
            PPO model parameters
        train_data : tuple
            Training data (prices, returns, tickers)
        val_data : Optional[tuple]
            Validation data (prices, returns, tickers)
        use_wandb : bool
            Whether to use Weights & Biases logging
        project_name : str
            Project name for wandb
        """
        self.config = config
        self.model_params = model_params
        self.train_data = train_data
        self.val_data = val_data
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        # Initialize environments
        self.train_env = self._create_env(train_data)
        self.eval_env = self._create_env(val_data) if val_data else None
        
        # Initialize model
        self.model = self._create_model()
        
    def _create_env(self, data: tuple) -> VecNormalize:
        """Create and wrap environment"""
        def make_env():
            env = FinancialEnv(self.config)
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
            **self.model_params
        )
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_path: Optional[str] = None
    ):
        """Train the model"""
        # Initialize wandb
        if self.use_wandb:
            wandb.init(project=self.project_name)
        
        # Create callbacks
        callbacks = [PortfolioCallback()]
        
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=n_eval_episodes
            )
            callbacks.append(eval_callback)
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
        # Save final model
        if save_path:
            self.model.save(os.path.join(save_path, "final_model"))
            self.train_env.save(os.path.join(save_path, "vec_normalize.pkl"))
        
        if self.use_wandb:
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
        # Initialize data manager with logging
        data_manager = DataManager()
        
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'PG', 'MA',
            'HD', 'BAC', 'CVX', 'KO', 'PFE', 'DIS', 'WMT', 'MRK', 'XOM', 'ORCL'
        ]

        periods = [
            ("2020-01-01", "2022-12-31"),
            ("2022-01-01", "2023-12-31")
        ]

        Xs = []
        ys = []
        selected_tickers = []

        for start_date, end_date in periods:
            num_assets = 10  # Change as needed
            print(f"\nLoading data for period {start_date} to {end_date} with {num_assets} assets")
            X, y, selected_tickers = data_manager.load_data(
                tickers=tickers,
                num_assets=num_assets,
                start_date=start_date,
                end_date=end_date
                
            )

            Xs.append(X)
            ys.append(y)
            selected_tickers.append(selected_tickers)

            print(f"Data shapes:")
            print(f"X: {X.shape}")  # Expected: (num_days, num_assets)
            print(f"y: {y.shape}")  # Expected: (num_days, num_assets)
            print(f"Number of assets: {len(selected_tickers)}")
            

        # Try loading training data
        print("Loading training data...")
        train_data = (Xs[0], ys[0], selected_tickers[0])
        
        print("Loading validation data...")
        val_data = (Xs[1], ys[1], selected_tickers[1])
        
        # Validate data shapes
        train_prices, train_returns, train_tickers = train_data
        print(f"\nTraining Data Shapes:")
        print(f"Prices: {train_prices.shape}")
        print(f"Returns: {train_returns.shape}")
        print(f"Tickers: {len(train_tickers)}")
        
        # Create config
        config = EnvironmentConfig(
            window_size=20,
            num_assets=len(train_tickers)
        )
        
        # Model parameters
        model_params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01
        }
        
        # Create trainer
        trainer = PortfolioTrainer(
            config=config,
            model_params=model_params,
            train_data=train_data,
            val_data=val_data,
            use_wandb=True
        )
        
        # Train model
        trainer.train(
            total_timesteps=1_000_000,
            eval_freq=10000,
            save_path="./models"
        )
        
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