import numpy as np
from typing import Dict, Optional, List
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from financial_env import FinancialEnv, EnvironmentConfig, ObservationType, RewardType

class EqualWeightBenchmark:
    """Computes equal weight strategy returns for comparison"""
    
    def __init__(self, returns: np.ndarray, num_assets: int):
        self.returns = returns
        self.num_assets = num_assets
        self.weights = np.ones(num_assets) / num_assets
    
    def get_returns(self, start_idx: int, end_idx: int) -> List[float]:
        """Calculate equal weight returns for a specific period"""
        period_returns = self.returns[start_idx:end_idx]
        portfolio_returns = np.sum(period_returns * self.weights, axis=1)
        return portfolio_returns.tolist()
    
    def get_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate performance metrics for a period"""
        if not returns:
            return {
                'mean_return': 0.0,
                'cumulative_return': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0
            }
            
        returns_array = np.array(returns)
        cumulative_return = np.prod(1 + returns_array) - 1
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe = mean_return / (std_return + 1e-6)
        
        return {
            'mean_return': float(mean_return),
            'cumulative_return': float(cumulative_return),
            'sharpe_ratio': float(sharpe),
            'volatility': float(std_return)
        }

class ImprovedPortfolioCallback(BaseCallback):
    """Callback for comparing agent performance with equal weight strategy"""
    
    def __init__(self, equal_weight_benchmark: EqualWeightBenchmark, window_size: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.equal_weight_benchmark = equal_weight_benchmark
        self.window_size = window_size
        self.episode_count = 0
        
        # Current episode tracking
        self.episode_returns = []
        self.equal_weight_returns = []
        
        # Running average tracking
        self.agent_metrics_history = []
        self.equal_metrics_history = []
        
    def _on_step(self) -> bool:
        """Process each step during training"""
        info = self.locals['infos'][0]  # Get info from first environment
        try:
            env = self.training_env.envs[0].env  # Get unwrapped env
        except Exception as e:
            env = self.training_env.envs[0]

        # Get current episode indices and returns
        current_idx = info.get('current_idx')
        episode_start = info.get('episode_start')
        k_day_returns = info.get('k_day_returns', [])
        
        if k_day_returns:
            # Get agent's portfolio return
            self.episode_returns.extend(k_day_returns)
            
            # Get equal weight returns for same period
            start_idx = current_idx - len(k_day_returns)
            equal_returns = self.equal_weight_benchmark.get_returns(
                start_idx=start_idx,
                end_idx=current_idx
            )
            self.equal_weight_returns.extend(equal_returns)
        
        # Check if episode is done
        if info.get("episode_end", None) is not None:
            # Calculate episode metrics
            agent_metrics = self.equal_weight_benchmark.get_metrics(self.episode_returns)
            equal_metrics = self.equal_weight_benchmark.get_metrics(self.equal_weight_returns)
            
            # Store metrics for running average
            self.agent_metrics_history.append(agent_metrics)
            self.equal_metrics_history.append(equal_metrics)
            
            # Keep only last window_size episodes
            if len(self.agent_metrics_history) > self.window_size:
                self.agent_metrics_history.pop(0)
                self.equal_metrics_history.pop(0)
            
            # Print running average every window_size episodes
            if self.episode_count % self.window_size == self.window_size - 1:
                self._print_running_average()
            
            # Reset episode tracking
            self.episode_count += 1
            self.episode_returns = []
            self.equal_weight_returns = []
        
        return True
    
    def _print_running_average(self):
        """Print running average of metrics over last window_size episodes"""
        start_episode = max(0, self.episode_count - self.window_size + 1)
        
        # Calculate average metrics
        avg_agent_metrics = {
            'mean_return': np.mean([m['mean_return'] for m in self.agent_metrics_history]),
            'cumulative_return': np.mean([m['cumulative_return'] for m in self.agent_metrics_history]),
            'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in self.agent_metrics_history]),
            'volatility': np.mean([m['volatility'] for m in self.agent_metrics_history])
        }
        
        avg_equal_metrics = {
            'mean_return': np.mean([m['mean_return'] for m in self.equal_metrics_history]),
            'cumulative_return': np.mean([m['cumulative_return'] for m in self.equal_metrics_history]),
            'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in self.equal_metrics_history]),
            'volatility': np.mean([m['volatility'] for m in self.equal_metrics_history])
        }
        
        print(f"\nRunning Average over Episodes {start_episode}-{self.episode_count}:")
        print(f"Agent Portfolio (avg over {len(self.agent_metrics_history)} episodes):")
        print(f"  Mean Return: {avg_agent_metrics['mean_return']:.4f}")
        print(f"  Cumulative Return: {avg_agent_metrics['cumulative_return']:.4f}")
        print(f"  Sharpe Ratio: {avg_agent_metrics['sharpe_ratio']:.4f}")
        print(f"  Volatility: {avg_agent_metrics['volatility']:.4f}")
        
        print(f"\nEqual Weight Portfolio:")
        print(f"  Mean Return: {avg_equal_metrics['mean_return']:.4f}")
        print(f"  Cumulative Return: {avg_equal_metrics['cumulative_return']:.4f}")
        print(f"  Sharpe Ratio: {avg_equal_metrics['sharpe_ratio']:.4f}")
        print(f"  Volatility: {avg_equal_metrics['volatility']:.4f}")
        
        # Print performance comparison
        return_diff = avg_agent_metrics['mean_return'] - avg_equal_metrics['mean_return']
        sharpe_diff = avg_agent_metrics['sharpe_ratio'] - avg_equal_metrics['sharpe_ratio']
        
        print(f"\nPerformance Difference (Agent - Equal Weight):")
        print(f"  Mean Return Diff: {return_diff:.4f}")
        print(f"  Sharpe Ratio Diff: {sharpe_diff:.4f}")
        print("-" * 50)

class ImprovedPortfolioTrainer:
    """Improved trainer implementation with proper episode structure"""
    
    def __init__(self, prices: np.ndarray, returns: np.ndarray, window_size: int = 20, 
                 prediction_days: int = 5, num_assets: int = 10):
        self.prices = prices
        self.returns = returns
        self.window_size = window_size
        self.prediction_days = prediction_days
        self.num_assets = num_assets
        
        # Initialize benchmark with just returns and num_assets
        self.equal_weight_benchmark = EqualWeightBenchmark(
            returns=returns,
            num_assets=num_assets
        )
        
        # Create environment and model
        self.env = self._create_env()
        self.model = self._create_model()
    
    def _create_env(self) -> VecNormalize:
        """Create and wrap the environment"""
        env_config = EnvironmentConfig(
            window_size=self.window_size,
            prediction_days=self.prediction_days,
            num_assets=self.num_assets,
            observation_types=list(ObservationType),
            reward_types=list(RewardType)
        )
        
        def make_env():
            return FinancialEnv(env_config, self.prices, self.returns)
        
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
        """Create PPO model with adjusted hyperparameters"""
        return PPO(
            "MultiInputPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.3,
            gae_lambda=0.95,
            verbose=1
        )
    
    def train(self, total_timesteps: int = 1000000):
        """Train the model with improved episode structure"""
        callback = ImprovedPortfolioCallback(
            equal_weight_benchmark=self.equal_weight_benchmark,
            window_size=5,
            verbose=1
        )
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
    
    def save(self, path: str):
        """Save model and environment normalization stats"""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "final_model"))
        self.env.save(os.path.join(path, "vec_normalize.pkl"))

def main():
    """Example usage"""
    # Load your data here
    from data_manager import DataManager
    
    data_manager = DataManager()
    prices, returns, tickers = data_manager.load_data(
        tickers="['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'DIS', 'PEP', 'ABEV', 'KO', 'PEP', 'ABEV', 'KO', 'PEP', 'ABEV', 'KO', 'PEP', 'ABEV', 'KO']",
        num_assets=20,
        start_date="2020-01-01",
        end_date="2022-12-31"
    )
    
    # Create and train the improved portfolio trainer
    trainer = ImprovedPortfolioTrainer(
        prices=prices,
        returns=returns,
        window_size=20,
        prediction_days=1,
        num_assets=20
    )
    
    # Train the model
    trainer.train(total_timesteps=1000000)
    
    # Save the trained model
    trainer.save("./improved_portfolio_model")

if __name__ == "__main__":
    main()