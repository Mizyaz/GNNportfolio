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
            reward_types=[RewardType.SHARPE]
        )
        
        env = FinancialEnv(config, prices, returns)
        obs, _ = env.reset()
        
        total_steps = 0
        portfolio_returns = []
        cumulative_return = 1.0
        
        # Track monthly performance
        monthly_returns = []
        current_month_returns = []
        last_month = None
        
        print(f"\nTesting {strategy_name} Strategy ({period}):")
        print(f"Initial observation shape: {obs['returns'].shape}")
        print(f"Total data points available: {len(returns)}")
        print(f"Expected steps (with {config.prediction_days}-day predictions): {len(returns) // config.prediction_days}")
        
        while True:
            # Get weights from strategy function
            weights = weight_func(env)
            
            if total_steps == 0:
                print(f"\nInitial portfolio weights:")
                print(f"Weights: {weights}")
                print(f"Weights sum: {np.sum(weights)}")
            
            # Take step in environment
            obs, reward, done, _, info = env.step(weights)
            
            # Record returns
            if 'portfolio_return' in info:
                step_return = info['portfolio_return']
                portfolio_returns.append(step_return)
                cumulative_return *= (1 + step_return)
                
                # Track monthly performance
                if 'current_date' in info:
                    current_month = info['current_date'].month
                    if last_month is None:
                        last_month = current_month
                    elif current_month != last_month:
                        if current_month_returns:
                            monthly_return = (np.prod([1 + r for r in current_month_returns]) - 1) * 100
                            monthly_returns.append(monthly_return)
                        current_month_returns = []
                        last_month = current_month
                    current_month_returns.append(step_return)
                
                # Print progress every 25 steps
                if total_steps % 25 == 0:
                    print(f"\nStep {total_steps}:")
                    print(f"Current return: {step_return:.4f}")
                    print(f"Cumulative return so far: {(cumulative_return - 1) * 100:.2f}%")
            else:
                print(f"Warning: No portfolio returns in info dict at step {total_steps}")
                print(f"Info keys available: {info.keys()}")
            
            total_steps += 1
            
            if done:
                break
        
        # Calculate performance metrics
        if portfolio_returns:
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            sharpe = mean_return / (std_return + 1e-6)
            total_return = (cumulative_return - 1) * 100  # Convert to percentage
            
            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + np.array(portfolio_returns))
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = np.min(drawdowns) * 100  # Convert to percentage
        else:
            mean_return = std_return = sharpe = total_return = max_drawdown = 0
        
        # Print final summary
        print(f"\n{strategy_name} Strategy ({period}) - Final Summary:")
        print(f"Total steps completed: {total_steps}")
        print(f"Total returns recorded: {len(portfolio_returns)}")
        if monthly_returns:
            print("\nMonthly Returns Summary:")
            print(f"Average Monthly Return: {np.mean(monthly_returns):.2f}%")
            print(f"Best Month: {np.max(monthly_returns):.2f}%")
            print(f"Worst Month: {np.min(monthly_returns):.2f}%")
        
        return {
            'total_return': total_return,
            'mean_return': mean_return * 100,  # Convert to percentage
            'std_return': std_return * 100,  # Convert to percentage
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_steps': total_steps,
            'num_returns': len(portfolio_returns),
            'monthly_returns': monthly_returns if monthly_returns else None
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
            if results['monthly_returns']:
                print("\nMonthly Performance:")
                print(f"Average Monthly Return: {np.mean(results['monthly_returns']):.2f}%")
                print(f"Best Month: {np.max(results['monthly_returns']):.2f}%")
                print(f"Worst Month: {np.min(results['monthly_returns']):.2f}%")
        
        print("\nTraining Period Results:")
        print_strategy_results(train_equal_weight, "Equal Weight", "Training")
        print_strategy_results(train_random_weight, "Random Weight", "Training")
        
        print("\nValidation Period Results:")
        print_strategy_results(val_equal_weight, "Equal Weight", "Validation")
        print_strategy_results(val_random_weight, "Random Weight", "Validation")
        
        print("\nValidation Complete!")

def main():
    validator = EnvironmentValidator(debug=True)
    validator.compare_strategies()

if __name__ == "__main__":
    main()