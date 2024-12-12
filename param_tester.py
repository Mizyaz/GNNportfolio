import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import logging
from financial_params import FinancialParameters, FinancialMetrics

@dataclass
class BLParameters:
    """Parameters for Black-Litterman model"""
    window_size: int = 20
    risk_free_rate: float = 0.02
    market_return: float = 0.08
    risk_aversion: float = 2.0
    tau: float = 0.05
    confidence_level: float = 0.95
    min_history_length: int = 5
    max_history_length: int = 252
    vol_lookback: int = 20
    view_lookback: int = 5
    regularization_factor: float = 1e-6
    use_exponential_weights: bool = True
    decay_factor: float = 0.94
    rebalance_frequency: int = 5

class ParamTester:
    def __init__(self, logger=None):
        """Initialize parameter tester with optional logger"""
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.test_results: List[Dict[str, Any]] = []
        self.financial_metrics = FinancialMetrics()
        
    def _setup_logging(self):
        """Setup logging if not already configured"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_data(self, returns_data: np.ndarray) -> Tuple[bool, str]:
        """Validate input data format and dimensions"""
        try:
            if not isinstance(returns_data, np.ndarray):
                return False, "Input must be numpy array"
            
            if returns_data.ndim != 2:
                return False, f"Expected 2D array, got {returns_data.ndim}D"
            
            if np.isnan(returns_data).any():
                return False, "Data contains NaN values"
            
            if np.isinf(returns_data).any():
                return False, "Data contains infinite values"
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}"

    def _test_black_litterman(self, 
                            returns_data: np.ndarray, 
                            params: BLParameters,
                            test_name: str) -> Dict[str, Any]:
        """Test Black-Litterman model with given parameters"""
        from black_litterman import BlackLittermanModel
        
        result = {
            "test_name": test_name,
            "success": False,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Validate data
            is_valid, message = self._validate_data(returns_data)
            if not is_valid:
                result["errors"].append(message)
                return result
            
            # Initialize model
            num_assets = returns_data.shape[1]
            bl_model = BlackLittermanModel(
                num_assets=num_assets,
                window_size=params.window_size,
                risk_free_rate=params.risk_free_rate,
                market_return=params.market_return,
                risk_aversion=params.risk_aversion,
                tau=params.tau,
                decay_factor=params.decay_factor,
                regularization_factor=params.regularization_factor,
                use_exponential_weights=params.use_exponential_weights
            )
            
            # Test prior calculation
            prior_returns, prior_cov = bl_model._calculate_prior(returns_data)
            if prior_returns is None or prior_cov is None:
                result["errors"].append("Prior calculation failed")
                return result
                
            # Test view incorporation
            posterior_returns, posterior_cov = bl_model._incorporate_views(
                prior_returns, prior_cov, returns_data
            )
            if posterior_returns is None or posterior_cov is None:
                result["errors"].append("View incorporation failed")
                return result
            
            # Test full optimization
            final_returns, final_cov = bl_model.optimize(returns_data)
            if final_returns is None or final_cov is None:
                result["errors"].append("Optimization failed")
                return result
            
            # Calculate test metrics
            result["metrics"] = {
                "mean_expected_return": float(np.mean(final_returns)),
                "portfolio_vol": float(np.sqrt(np.diag(final_cov).mean())),
                "condition_number": float(np.linalg.cond(final_cov))
            }
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(f"Test error: {str(e)}")
            
        return result

    def _test_financial_metrics(self, 
                              prices: np.ndarray,
                              returns: np.ndarray) -> Dict[str, Any]:
        """Test financial metrics calculation"""
        result = {
            "success": False,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Calculate trend metrics
            trend_metrics = self.financial_metrics.calculate_trend_metrics(prices)
            
            # Calculate diversification metrics
            div_metrics = self.financial_metrics.calculate_diversification_metrics(returns)
            
            # Calculate technical indicators
            tech_metrics = self.financial_metrics.calculate_technical_indicators(prices)
            
            # Combine all metrics
            result["metrics"].update({
                "trend_strength_mean": float(np.mean(trend_metrics["trend_strength"])),
                "momentum_mean": float(np.mean(trend_metrics["momentum"])),
                "diversification_ratio": float(div_metrics["diversification_ratio"]),
                "effective_n": float(div_metrics["effective_n"]),
                "avg_rsi": float(np.mean(tech_metrics["rsi"])),
                "avg_volatility": float(np.mean(tech_metrics["volatility"]))
            })
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(f"Financial metrics calculation failed: {str(e)}")
        
        return result

    def run_tests(self, 
                 returns_data: np.ndarray, 
                 param_sets: List[Tuple[BLParameters, str]]) -> pd.DataFrame:
        """
        Run tests with multiple parameter sets
        
        Parameters:
        -----------
        returns_data : np.ndarray
            Returns data matrix (time x assets)
        param_sets : List[Tuple[BLParameters, str]]
            List of (parameters, test_name) tuples to test
            
        Returns:
        --------
        pd.DataFrame
            Test results summary
        """
        self._setup_logging()
        self.test_results = []
        
        for params, test_name in param_sets:
            self.logger.info(f"Running test: {test_name}")
            result = self._test_black_litterman(returns_data, params, test_name)
            self.test_results.append(result)
            
            if result["success"]:
                self.logger.info(f"Test {test_name} succeeded")
                self.logger.info(f"Metrics: {result['metrics']}")
            else:
                self.logger.error(f"Test {test_name} failed")
                for error in result["errors"]:
                    self.logger.error(f"Error: {error}")
        
        # Add financial metrics testing
        prices = np.exp(np.cumsum(returns_data, axis=0))  # Convert returns to prices
        financial_result = self._test_financial_metrics(prices, returns_data)
        
        if financial_result["success"]:
            self.logger.info("Financial metrics calculation succeeded")
            self.logger.info(f"Metrics: {financial_result['metrics']}")
        else:
            self.logger.error("Financial metrics calculation failed")
            for error in financial_result["errors"]:
                self.logger.error(f"Error: {error}")
        
        return self._create_summary()

    def _create_summary(self) -> pd.DataFrame:
        """Create summary DataFrame of test results"""
        summary_data = []
        
        for result in self.test_results:
            row = {
                "Test Name": result["test_name"],
                "Success": result["success"],
                "Error Count": len(result["errors"]),
                "Warning Count": len(result["warnings"])
            }
            
            if result["success"] and "metrics" in result:
                row.update(result["metrics"])
                
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)

def main():
    """Example usage"""
    # Create sample data
    np.random.seed(42)
    returns_data = np.random.randn(100, 10) * 0.01  # 100 days, 10 assets
    
    # Create parameter sets
    param_sets = [
        (BLParameters(window_size=20, decay_factor=0.94), "Default Parameters"),
        (BLParameters(window_size=60, decay_factor=0.98), "Long Window"),
        (BLParameters(window_size=10, decay_factor=0.90), "Short Window")
    ]
    
    # Run tests
    tester = ParamTester()
    results = tester.run_tests(returns_data, param_sets)
    
    print("\nTest Results Summary:")
    print(results)

if __name__ == "__main__":
    main() 