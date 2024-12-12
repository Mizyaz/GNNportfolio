import logging
import numpy as np
from financial_env import FinancialEnv, EnvironmentConfig, ObservationType, RewardType
from data_manager import DataManager
from typing import Dict, List, Tuple
import pandas as pd

class EnvironmentValidator:
    """Validator for FinancialEnv with various edge cases"""
    
    def __init__(self, logger=None, debug=True):
        self.logger = logger or logging.getLogger(__name__)
        self.debug = debug
        self._setup_logging()
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Default configuration
        self.train_period = ("2020-01-01", "2022-12-31")
        self.val_period = ("2022-01-01", "2023-12-31")
        self.num_assets = 5
        self.tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'PG', 'MA'
        ]
        
        # Load data for both periods
        self._load_data()
        
    def _load_data(self):
        """Load data for both training and validation periods"""
        print("\nLoading training data...")
        train_data = self.data_manager.load_data(
            tickers=self.tickers,
            num_assets=self.num_assets,
            start_date=self.train_period[0],
            end_date=self.train_period[1]
        )
        self.train_prices, self.train_returns, self.selected_tickers = train_data
        
        print("\nLoading validation data...")
        val_data = self.data_manager.load_data(
            tickers=self.selected_tickers,  # Use same tickers as training
            num_assets=self.num_assets,
            start_date=self.val_period[0],
            end_date=self.val_period[1]
        )
        self.val_prices, self.val_returns, _ = val_data
        
    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if not self.debug else logging.DEBUG)
    
    def _test_strategy_performance(self, strategy_name: str, weight_func, prices: np.ndarray, returns: np.ndarray, period: str) -> Dict[str, float]:
        """Test performance of a given portfolio weighting strategy"""
        config = EnvironmentConfig(
            window_size=20,
            prediction_days=5,
            num_assets=self.num_assets,
            observation_types=[ObservationType.RETURNS],
            reward_types=[RewardType.SHARPE],
            max_steps=30  # Limit episode length
        )
        
        env = FinancialEnv(config, prices, returns)
        
        episode_metrics = []
        num_episodes = 10
        
        print(f"\nTesting {strategy_name} Strategy ({period}):")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_returns = []
            
            print(f"\nEpisode {episode + 1}:")
            print(f"Start index: {env.episode_start_idx}")
            
            while not done:
                weights = weight_func(env)
                obs, reward, done, _, info = env.step(weights)
                episode_returns.extend(info.get('k_day_returns', []))
            
            print(f"End index: {env.episode_end_idx}")
            print(f"Episode length: {len(episode_returns)} days")
            print(f"Cumulative return: {info['episode_cumulative_return']:.4f}")
            print(f"Sharpe ratio: {info['episode_sharpe']:.4f}")
            
            episode_metrics.append(info)
        
        # Calculate aggregate statistics
        cumulative_returns = [m['episode_cumulative_return'] for m in episode_metrics]
        sharpe_ratios = [m['episode_sharpe'] for m in episode_metrics]
        
        return {
            'mean_cumulative_return': np.mean(cumulative_returns),
            'std_cumulative_return': np.std(cumulative_returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'num_episodes': num_episodes,
            "total_return": np.mean(cumulative_returns),
            "mean_return": np.mean(cumulative_returns),
            "std_return": np.std(cumulative_returns),
            "sharpe_ratio": np.mean(sharpe_ratios),
            "max_drawdown": np.min(cumulative_returns),
            "num_steps": num_episodes,
            "num_returns": num_episodes,
            "quarter_returns": [np.mean(cumulative_returns[i:i+3]) for i in range(0, len(cumulative_returns), 3)]
        }
    
    def compare_strategies(self):
        """Compare equal weight vs random weight strategies on both periods"""
        # Print data summary for both periods
        print("\nData Summary:")
        print("\nTraining Period:")
        print(f"Period: {self.train_period[0]} to {self.train_period[1]}")
        print(f"Number of days: {len(self.train_returns)}")
        print(f"Number of assets: {self.num_assets}")
        print(f"Selected tickers: {self.selected_tickers}")
        print("Return statistics:")
        print(f"Mean returns per asset: {np.mean(self.train_returns, axis=0)}")
        print(f"Std returns per asset: {np.std(self.train_returns, axis=0)}")
        print(f"Price ranges per asset: {np.min(self.train_prices, axis=0)} - {np.max(self.train_prices, axis=0)}")
        
        print("\nValidation Period:")
        print(f"Period: {self.val_period[0]} to {self.val_period[1]}")
        print(f"Number of days: {len(self.val_returns)}")
        print("Return statistics:")
        print(f"Mean returns per asset: {np.mean(self.val_returns, axis=0)}")
        print(f"Std returns per asset: {np.std(self.val_returns, axis=0)}")
        print(f"Price ranges per asset: {np.min(self.val_prices, axis=0)} - {np.max(self.val_prices, axis=0)}")
        
        print("\nRunning Strategy Comparison Test...")
        
        # Test strategies on training period
        train_equal_weight = self._test_strategy_performance(
            "Equal Weight",
            lambda env: np.ones(env.config.num_assets) / env.config.num_assets,
            self.train_prices,
            self.train_returns,
            "Training"
        )
        
        train_random_weight = self._test_strategy_performance(
            "Random Weight",
            lambda env: np.random.dirichlet(np.ones(env.config.num_assets)),
            self.train_prices,
            self.train_returns,
            "Training"
        )
        
        # Test strategies on validation period
        val_equal_weight = self._test_strategy_performance(
            "Equal Weight",
            lambda env: np.ones(env.config.num_assets) / env.config.num_assets,
            self.val_prices,
            self.val_returns,
            "Validation"
        )
        
        val_random_weight = self._test_strategy_performance(
            "Random Weight",
            lambda env: np.random.dirichlet(np.ones(env.config.num_assets)),
            self.val_prices,
            self.val_returns,
            "Validation"
        )
        
        # Print comparison results
        print("\nStrategy Comparison Results:")
        print("-" * 50)
        
        def print_strategy_results(results, strategy_name, period):
            print(f"\n{strategy_name} Strategy ({period}):")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Mean Daily Return: {results['mean_return']:.4f}%")
            print(f"Daily Return Std: {results['std_return']:.4f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Number of Steps: {results['num_steps']}")
            print(f"Number of Returns: {results['num_returns']}")
            if results['quarter_returns']:
                print("\nQuarterly Performance:")
                for i, qret in enumerate(results['quarter_returns'], 1):
                    print(f"Q{i}: {qret:.2f}%")
        
        print("\nTraining Period Results:")
        print("-" * 30)
        print_strategy_results(train_equal_weight, "Equal Weight", "Training")
        print("\nVS\n")
        print_strategy_results(train_random_weight, "Random Weight", "Training")
        
        print("\nValidation Period Results:")
        print("-" * 30)
        print_strategy_results(val_equal_weight, "Equal Weight", "Validation")
        print("\nVS\n")
        print_strategy_results(val_random_weight, "Random Weight", "Validation")
        
        # Print strategy comparison summary
        print("\nStrategy Comparison Summary:")
        print("-" * 30)
        print("\nTraining Period:")
        print(f"Equal Weight vs Random Weight:")
        print(f"Total Return Difference: {train_equal_weight['total_return'] - train_random_weight['total_return']:.2f}%")
        print(f"Sharpe Ratio Difference: {train_equal_weight['sharpe_ratio'] - train_random_weight['sharpe_ratio']:.2f}")
        
        print("\nValidation Period:")
        print(f"Equal Weight vs Random Weight:")
        print(f"Total Return Difference: {val_equal_weight['total_return'] - val_random_weight['total_return']:.2f}%")
        print(f"Sharpe Ratio Difference: {val_equal_weight['sharpe_ratio'] - val_random_weight['sharpe_ratio']:.2f}")
        
        print("\nValidation Complete!")

def main():
    validator = EnvironmentValidator(debug=True)
    validator.compare_strategies()

if __name__ == "__main__":
    main()