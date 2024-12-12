import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import yfinance as yf
import pandas as pd
import os
import pickle
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from black_litterman import BlackLittermanModel

# Suppress TensorFlow messages if not needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_and_save_data(num_assets=10, start_date="2020-01-01", end_date="2022-12-31", force_reload=False):
    """
    Load or download financial data

    Parameters:
    -----------
    num_assets : int
        Number of assets to include
    start_date : str
        Start date for data collection
    end_date : str
        End date for data collection
    force_reload : bool
        If True, force reload data even if it exists
    """
    # Create data directory if it doesn't exist
    os.makedirs('rl_data', exist_ok=True)

    # Generate filename based on parameters
    filename_prefix = f"{start_date.replace('-', '_')}_{end_date.split('-')[0]}"
    x_filename = f'rl_data/x_train_{filename_prefix}.pkl'
    y_filename = f'rl_data/y_train_{filename_prefix}.pkl'
    tickers_filename = f'rl_data/tickers_{filename_prefix}.pkl'

    # Check if processed data already exists
    if not force_reload and all(os.path.exists(fname) for fname in [x_filename, y_filename, tickers_filename]):
        print("Loading existing data...")
        with open(x_filename, 'rb') as f:
            x_train = pickle.load(f)
        with open(y_filename, 'rb') as f:
            y_train = pickle.load(f)
        with open(tickers_filename, 'rb') as f:
            saved_tickers = pickle.load(f)

        # Check if we have enough assets
        if len(saved_tickers) >= num_assets:
            print(f"Using existing data with {len(saved_tickers)} assets")
            return x_train, y_train, saved_tickers[:num_assets]
        else:
            print(f"Existing data has only {len(saved_tickers)} assets, need {num_assets}. Reloading...")

    print(f"Downloading data for {num_assets} assets from {start_date} to {end_date}...")
    # Read requested number of SP500 tickers
    with open('sp500tickers.txt', 'r') as f:
        sp500_tickers = [line.strip() for line in f.readlines()][:num_assets]

    # Download data for all tickers
    all_data = {}
    first_valid_data = None

    for ticker in sp500_tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                close_prices = stock_data['Close'].values.flatten()
                all_data[ticker] = close_prices
                if first_valid_data is None:
                    first_valid_data = stock_data.index
                elif len(close_prices) != len(first_valid_data):
                    print(f"Skipping {ticker} due to length mismatch")
                    del all_data[ticker]
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    if first_valid_data is not None:
        df = pd.DataFrame(all_data, index=first_valid_data)
        df = df.dropna(axis=1)  # Drop columns with any missing values

        if df.empty:
            raise ValueError("No valid data remaining after cleaning")

        # Create features and targets
        x_train = df.values[:-1]  # All data except last day
        y_train = (df.values[1:] - df.values[:-1]) / df.values[:-1]  # Daily returns

        # Save processed data with timestamp
        with open(x_filename, 'wb') as f:
            pickle.dump(x_train, f)
        with open(y_filename, 'wb') as f:
            pickle.dump(y_train, f)
        with open(tickers_filename, 'wb') as f:
            pickle.dump(df.columns.tolist(), f)

        return x_train, y_train, df.columns.tolist()
    else:
        raise ValueError("No valid data downloaded for any ticker")


class PortfolioAllocationEnv(gym.Env):
    def __init__(self, x_train, y_train, tickers, num_features=34, transaction_cost=0.001, 
                 risk_free_rate=0.02, market_return=0.08, risk_aversion=2.0):
        super(PortfolioAllocationEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.num_features = num_features
        self.transaction_cost = transaction_cost

        # Initialize previous weights for transaction cost calculation
        self.previous_weights = np.ones(self.num_assets) / self.num_assets
        # Initialize returns window for Sharpe ratio calculation
        self.returns_window = []

        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features, self.num_assets),
            dtype=np.float32
        )

        # Define action space
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        self.state = None
        self.current_step = 0
        self.max_steps = len(self.x_train) - 1

        # Add Black-Litterman parameters
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.risk_aversion = risk_aversion
        self.returns_history = []
        self.weights_history = []

        # Initialize Black-Litterman model
        self.bl_model = BlackLittermanModel(
            num_assets=len(tickers),
            window_size=20,
            risk_free_rate=risk_free_rate,
            market_return=market_return,
            risk_aversion=risk_aversion
        )

    def _create_state(self):
        if self.current_step >= self.max_steps:
            return np.zeros((self.num_features, self.num_assets), dtype=np.float32)

        state = np.zeros((self.num_features, self.num_assets))
        feature_idx = 0

        # Recent returns (last 20 days)
        lookback = min(20, self.current_step + 1)
        returns_window = self.y_train[max(0, self.current_step - lookback + 1):self.current_step + 1]
        state[feature_idx:feature_idx + returns_window.shape[0], :] = returns_window
        feature_idx += 20

        # Current prices
        if feature_idx < self.num_features:
            state[feature_idx] = self.x_train[self.current_step]
        feature_idx += 1

        # Moving averages
        for window in [5, 10, 20]:
            if feature_idx < self.num_features and self.current_step >= window:
                ma = np.mean(self.y_train[self.current_step - window + 1:self.current_step + 1], axis=0)
                state[feature_idx] = ma
            feature_idx += 1

        # Volatility
        if feature_idx < self.num_features and self.current_step >= 20:
            vol = np.std(self.y_train[self.current_step - 20:self.current_step + 1], axis=0)
            state[feature_idx] = vol

        return state.astype(np.float32)

    def calculate_reward(self, portfolio_return, action):
        """Enhanced reward function incorporating Black-Litterman insights"""
        try:
            # Store returns history
            self.returns_history.append(portfolio_return)
            self.weights_history.append(action)
            
            # Get recent returns for Black-Litterman model
            recent_returns = np.array(self.returns_history[-20:])
            if len(recent_returns) > 1:
                # Get Black-Litterman parameters
                expected_returns, cov_matrix = self.bl_model.optimize(recent_returns)
                
                # Calculate portfolio metrics
                portfolio_vol = np.sqrt(np.maximum(action.T @ cov_matrix @ action, 0))
                portfolio_vol = max(portfolio_vol, 1e-6)
                
                # Expected return based on Black-Litterman
                expected_portfolio_return = np.dot(action, expected_returns)
                
                # Calculate reward components
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
                utility = portfolio_return - 0.5 * self.risk_aversion * (portfolio_vol ** 2)
                div_penalty = -0.5 * np.sum(action ** 2)  # Diversification penalty
                tracking_error = np.sum((action - 1/self.num_assets) ** 2)
                
                # Combine components with proper scaling
                reward = (0.4 * np.clip(sharpe_ratio, -5, 5) +
                         0.3 * np.clip(utility, -5, 5) +
                         0.2 * div_penalty +
                         0.1 * (-tracking_error))
                
                return float(np.clip(reward, -10, 10))
            
            return float(portfolio_return)  # Default to simple return when insufficient history
            
        except Exception as e:
            print(f"Warning: Error in reward calculation: {e}")
            return -1.0

    def step(self, action):
        # Ensure action is a numpy array
        action = np.array(action).flatten()

        # Normalize action to ensure portfolio weights sum to 1
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.num_assets) / self.num_assets

        # Get returns for the next 5 days
        end_step = self.current_step + 5
        if end_step >= self.max_steps:
            end_step = self.max_steps
            done = True
        else:
            done = False

        returns = self.y_train[self.current_step:end_step].mean(axis=0)  # Average return over next 5 days

        # Calculate portfolio return
        portfolio_return = np.dot(action, returns)

        # Calculate transaction costs
        turnover = np.abs(action - self.previous_weights).sum()
        transaction_cost = turnover * self.transaction_cost

        # Update returns window
        self.returns_window.append(portfolio_return - transaction_cost)
        if len(self.returns_window) > 20:  # Keep last 20 periods for Sharpe calculation
            self.returns_window.pop(0)

        # Replace simple reward calculation with enhanced version
        reward = self.calculate_reward(portfolio_return - transaction_cost, action)

        # Update state and previous weights
        self.current_step = end_step
        self.state = self._create_state()
        self.previous_weights = action.copy()  # Make a copy to avoid reference issues

        # Check if episode is done
        truncated = False  # You can define truncation conditions if needed

        # Calculate volatility for info
        volatility = np.std(self.returns_window) if len(self.returns_window) > 1 else 0.0

        info = {
            'portfolio_return': float(portfolio_return),
            'transaction_cost': float(transaction_cost),
            'volatility': float(volatility),
            'reward': float(reward)
        }

        return self.state, float(reward), done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._create_state()
        self.previous_weights = np.ones(self.num_assets) / self.num_assets
        self.returns_window = []
        return self.state, {}


def compare_strategies(model, train_env, test_env, num_episodes=10):
    """Compare RL model with equal weight strategy on both train and test sets"""

    def evaluate_equal_weight(env):
        portfolio_returns = []
        obs, _ = env.reset()
        done = False
        equal_weights = np.ones(env.num_assets) / env.num_assets

        while not done:
            action = equal_weights
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            portfolio_returns.append(info['portfolio_return'])

        return np.mean(portfolio_returns), np.std(portfolio_returns)

    def evaluate_rl(env, model):
        all_portfolio_returns = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_returns = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_returns.append(info['portfolio_return'])

            all_portfolio_returns.append(np.mean(episode_returns))

        return np.mean(all_portfolio_returns), np.std(all_portfolio_returns)

    # Evaluate on training set
    print("\nEvaluating on training set:")
    rl_train_mean, rl_train_std = evaluate_rl(train_env, model)
    eq_train_mean, eq_train_std = evaluate_equal_weight(train_env)

    # Evaluate on test set
    print("\nEvaluating on test set:")
    rl_test_mean, rl_test_std = evaluate_rl(test_env, model)
    eq_test_mean, eq_test_std = evaluate_equal_weight(test_env)

    results = {
        'Training Set': {
            'RL Strategy': f"{rl_train_mean:.4f} ± {rl_train_std:.4f}",
            'Equal Weight': f"{eq_train_mean:.4f} ± {eq_train_std:.4f}",
            'Outperformance': f"{(rl_train_mean - eq_train_mean):.4f}"
        },
        'Test Set': {
            'RL Strategy': f"{rl_test_mean:.4f} ± {rl_test_std:.4f}",
            'Equal Weight': f"{eq_test_mean:.4f} ± {eq_test_std:.4f}",
            'Outperformance': f"{(rl_test_mean - eq_test_mean):.4f}"
        }
    }

    return pd.DataFrame(results)

def visualize_results(model, env, title="Portfolio Performance"):
    """Visualize portfolio performance and allocation"""
    obs, _ = env.reset()
    done = False
    portfolio_values = [1.0]
    weight_history = []
    returns_history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store results
        portfolio_values.append(portfolio_values[-1] * (1 + info['portfolio_return']))
        weight_history.append(action)
        returns_history.append(info['portfolio_return'])
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot portfolio value
    ax1.plot(portfolio_values, label='Portfolio Value')
    ax1.set_title(f'{title} - Portfolio Value Over Time')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot weight allocation heatmap
    weight_history = np.array(weight_history)
    sns.heatmap(weight_history.T, ax=ax2, cmap='YlOrRd')
    ax2.set_title('Portfolio Weights Over Time')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Assets')
    
    # Plot returns distribution
    sns.histplot(returns_history, kde=True, ax=ax3)
    ax3.set_title('Returns Distribution')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def main():
    # Load train and test data
    x_train, y_train, tickers = load_and_save_data(
        num_assets=10,
        start_date="2020-01-01",
        end_date="2022-12-31"
    )

    x_test, y_test, test_tickers = load_and_save_data(
        num_assets=10,
        start_date="2022-01-01",
        end_date="2023-12-31"
    )

    # Create environments
    train_env = PortfolioAllocationEnv(
        x_train=x_train,
        y_train=y_train,
        tickers=tickers,
        num_features=34
    )

    test_env = PortfolioAllocationEnv(
        x_train=x_test,
        y_train=y_test,
        tickers=test_tickers,
        num_features=34
    )

    # Initialize model with better hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )

    # Train for more steps
    model.learn(total_timesteps=10000)

    # Visualize results
    print("\nGenerating visualizations...")
    train_fig = visualize_results(model, train_env, "Training Set")
    test_fig = visualize_results(model, test_env, "Test Set")
    
    # Save figures
    train_fig.savefig('training_results.png')
    test_fig.savefig('testing_results.png')
    
    # Compare strategies
    results = compare_strategies(model, train_env, test_env)
    print("\nStrategy Comparison:")
    print(results)


if __name__ == "__main__":
    main()
