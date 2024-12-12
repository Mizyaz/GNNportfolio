import numpy as np
from typing import Dict, List, Tuple
import logging
from financial_env import FinancialEnv, EnvironmentConfig, ObservationType, RewardType

class EnvironmentValidator:
    """Validator for FinancialEnv with various edge cases"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def run_validation_suite(self) -> Dict[str, bool]:
        """Run all validation tests"""
        results = {}
        
        test_cases = [
            self._test_basic_functionality,
            self._test_edge_case_zero_weights,
            self._test_edge_case_negative_returns,
            self._test_edge_case_missing_data,
            self._test_edge_case_extreme_values,
            self._test_observation_types,
            self._test_reward_types,
            self._test_sequential_steps
        ]
        
        for test in test_cases:
            test_name = test.__name__
            try:
                test()
                results[test_name] = True
                self.logger.info(f"✓ {test_name} passed")
            except Exception as e:
                results[test_name] = False
                self.logger.error(f"✗ {test_name} failed: {str(e)}")
        
        return results
    
    def _test_basic_functionality(self):
        """Test basic environment functionality"""
        config = EnvironmentConfig(
            window_size=10,
            num_assets=5,
            observation_types=[ObservationType.RETURNS],
            reward_types=[RewardType.SHARPE]
        )
        env = FinancialEnv(config)
        
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert isinstance(reward, float), "Reward should be a float"
        assert isinstance(done, bool), "Done should be a boolean"
        
    def _test_edge_case_zero_weights(self):
        """Test behavior with zero weights"""
        config = EnvironmentConfig(window_size=5, num_assets=3)
        env = FinancialEnv(config)
        
        env.reset()
        action = np.zeros(env.config.num_assets)
        next_obs, reward, done, truncated, info = env.step(action)
        
        assert np.allclose(np.sum(info['weights']), 1.0), "Weights should sum to 1"
        
    def _test_edge_case_negative_returns(self):
        """Test behavior with negative returns"""
        config = EnvironmentConfig(window_size=5, num_assets=3)
        env = FinancialEnv(config)
        
        env.reset()
        env.returns_history = -np.ones_like(env.returns_history)
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        assert np.isfinite(reward), "Reward should be finite even with negative returns"
        
    def _test_edge_case_missing_data(self):
        """Test behavior with missing data"""
        config = EnvironmentConfig(window_size=5, num_assets=3)
        env = FinancialEnv(config)
        
        env.reset()
        env.returns_history[2, 1] = np.nan
        try:
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            assert False, "Should raise error on NaN values"
        except:
            assert True
            
    def _test_edge_case_extreme_values(self):
        """Test behavior with extreme values"""
        config = EnvironmentConfig(window_size=5, num_assets=3)
        env = FinancialEnv(config)
        
        env.reset()
        env.returns_history = np.ones_like(env.returns_history) * 1e6
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        assert np.isfinite(reward), "Reward should be finite with extreme values"
        
    def _test_observation_types(self):
        """Test all observation types"""
        config = EnvironmentConfig(
            window_size=10,
            num_assets=5,
            observation_types=list(ObservationType)
        )
        env = FinancialEnv(config)
        
        obs, _ = env.reset()
        for obs_type in ObservationType:
            assert obs_type.value in obs, f"Missing observation type: {obs_type.value}"
            
    def _test_reward_types(self):
        """Test all reward types"""
        config = EnvironmentConfig(
            window_size=10,
            num_assets=5,
            reward_types=list(RewardType)
        )
        env = FinancialEnv(config)
        
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        
        for reward_type in RewardType:
            assert reward_type.value in info['reward_components']
            
    def _test_sequential_steps(self):
        """Test multiple sequential steps"""
        config = EnvironmentConfig(window_size=5, num_assets=3)
        env = FinancialEnv(config)
        
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done:
                env.reset()

def run_quick_test():
    """Run a quick test of the environment"""
    config = EnvironmentConfig(
        window_size=20,
        num_assets=10,
        observation_types=[
            ObservationType.RETURNS,
            ObservationType.TECHNICAL
        ],
        reward_types=[
            RewardType.SHARPE,
            RewardType.DIVERSIFICATION
        ]
    )
    
    env = FinancialEnv(config)
    obs, _ = env.reset()
    
    print("\nQuick Test Results:")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    
    print("\nStep Results:")
    print(f"Reward: {reward:.4f}")
    print("Portfolio Weights:", info['weights'])
    print("Reward Components:", info['reward_components'])

if __name__ == "__main__":
    validator = EnvironmentValidator()
    results = validator.run_validation_suite()
    
    print("\nValidation Results Summary:")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {sum(results.values())}")
    print(f"Failed: {len(results) - sum(results.values())}")
    
    run_quick_test() 