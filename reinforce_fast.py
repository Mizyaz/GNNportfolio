import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import pickle
import os

# Step 1: Load and prepare data
def download_data(tickers, start_date, end_date):
    price_df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            price_df[ticker] = data['Adj Close']
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return price_df

def calculate_returns(price_data):
    return price_data.pct_change().dropna()

def select_top_k_assets(returns_df, K):
    # Select K assets with highest volatility (standard deviation of returns)
    volatility = returns_df.std()
    top_k = volatility.sort_values(ascending=False).head(K).index.tolist()
    print(f"Selected top {K} assets based on volatility.")
    return top_k

# Step 2: Define the custom Gym environment
class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, returns_df, selected_assets, window_size=20):
        super(TradingEnv, self).__init__()
        
        self.returns_df = returns_df[selected_assets]
        self.tickers = selected_assets
        self.window_size = window_size
        self.current_step = window_size
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, len(self.tickers)), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(self.tickers),), dtype=np.float32
        )
        
        self.portfolio_weights = np.ones(len(self.tickers)) / len(self.tickers)
        self.done = False
    
    def reset(self, seed=None):
        self.current_step = self.window_size
        self.portfolio_weights = np.ones(len(self.tickers)) / len(self.tickers)
        self.done = False
        return self._get_observation(), {}
    
    def step(self, action):
        action = np.clip(action, 0, 1)
        if action.sum() == 0:
            self.portfolio_weights = np.ones(len(self.tickers)) / len(self.tickers)
        else:
            self.portfolio_weights = action / action.sum()
        
        next_returns = self.returns_df.iloc[self.current_step].values
        reward = np.dot(self.portfolio_weights, next_returns)
        
        self.current_step += 1
        if self.current_step >= len(self.returns_df):
            self.done = True
        
        return self._get_observation(), reward, self.done, False, {}
    
    def _get_observation(self):
        return self.returns_df.iloc[self.current_step - self.window_size:self.current_step].values
    
    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Portfolio Weights: {self.portfolio_weights}")

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved object to {filename}")

def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded object from {filename}")
    return obj

def plot_cumulative_returns(strategy_returns, benchmark_returns, title='Cumulative Returns'):
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_returns, label='PPO Strategy')
    plt.plot(benchmark_returns, label='Equal-Weighted Portfolio')
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='RL Portfolio Optimization with Dimensionality Reduction')
    parser.add_argument('--train', action='store_true', help='Train the PPO agent')
    parser.add_argument('--test', action='store_true', help='Test the PPO agent')
    parser.add_argument('--k_assets', type=int, default=10, help='Number of top assets to select')
    parser.add_argument('--window_size', type=int, default=20, help='Number of previous days used for observation')
    parser.add_argument('--timesteps', type=int, default=200000, help='Number of timesteps for training')
    args = parser.parse_args()

    # set default values
    if not args.train and not args.test:
        args.train = True
        args.test = True
    if not args.k_assets:
        args.k_assets = 100
    if not args.window_size:
        args.window_size = 30
    if not args.timesteps:
        args.timesteps = 1000000

    # Hyperparameters
    K_ASSETS = args.k_assets
    WINDOW_SIZE = args.window_size
    TRAINING_TIMESTEPS = args.timesteps

    # File paths
    PRICE_DATA_FILE = 'price_df.pkl'
    REDUCED_DATA_FILE = f'./data/reduced_returns_k{K_ASSETS}.pkl'
    MODEL_FILE = f'./models/ppo_trading_agent_k{K_ASSETS}'

    # Load S&P 500 tickers
    if not os.path.exists('sp500tickers.txt'):
        print("Error: 'sp500tickers.txt' not found.")
        return

    with open('sp500tickers.txt', 'r') as f:
        tickers = f.read().splitlines()
    
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    start_date = '2021-12-31'
    end_date = '2023-12-31'
    
    # Download historical data
    if os.path.exists(PRICE_DATA_FILE):
        price_data = load_pickle(PRICE_DATA_FILE)
    else:
        print("Downloading historical data...")
        price_data = download_data(tickers, start_date, end_date)
        save_pickle(price_data, PRICE_DATA_FILE)
    
    # Calculate daily returns
    returns_df = calculate_returns(price_data)
    
    # Select top K assets
    if os.path.exists(REDUCED_DATA_FILE):
        selected_assets = load_pickle(REDUCED_DATA_FILE)['selected_assets']
        train_returns = load_pickle(REDUCED_DATA_FILE)['train_returns']
        test_returns = load_pickle(REDUCED_DATA_FILE)['test_returns']
    else:
        selected_assets = select_top_k_assets(returns_df, K_ASSETS)
        # Split into training and testing datasets
        train_returns = returns_df['2022-01-01':'2022-12-31'][selected_assets]
        test_returns = returns_df['2023-01-01':'2023-12-31'][selected_assets]
        # Save reduced data
        reduced_data = {
            'selected_assets': selected_assets,
            'train_returns': train_returns,
            'test_returns': test_returns
        }
        save_pickle(reduced_data, REDUCED_DATA_FILE)
    
    if args.train:
        # Step 3: Train the PPO agent
        print("Training the PPO agent...")
        
        # Create the training environment
        train_env = lambda: TradingEnv(train_returns, selected_assets, window_size=WINDOW_SIZE)
        vec_env = make_vec_env(train_env, n_envs=4)
        
        # Initialize the PPO model
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99
        )
        
        # Train the agent
        model.learn(total_timesteps=TRAINING_TIMESTEPS)
        
        # Save the model
        model_path = f"{MODEL_FILE}.zip"
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    if args.test:
        # Step 4: Test the PPO agent
        print("Testing the PPO agent...")
        
        # Load the model
        model_path = f"{MODEL_FILE}.zip"
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Please train the model first.")
            return
        model = PPO.load(model_path)
        
        # Create the testing environment
        test_env = TradingEnv(test_returns, selected_assets, window_size=WINDOW_SIZE)
        obs, _ = test_env.reset()
        done = False
        cumulative_reward = 0
        daily_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = test_env.step(action)
            cumulative_reward += reward
            daily_rewards.append(reward)
            # Uncomment the next line to see step-by-step actions
            # test_env.render()
        
        print(f"Cumulative Reward: {cumulative_reward:.4f}")
        
        # Step 5: Evaluate performance
        print("Evaluating performance...")
        
        # Calculate cumulative returns
        strategy_cumulative_returns = np.cumprod([1 + r for r in daily_rewards])
        
        # Benchmark: Equal-weighted portfolio
        equal_weights = np.ones(len(selected_assets)) / len(selected_assets)
        benchmark_returns = test_returns.mean(axis=1).values
        benchmark_cumulative_returns = np.cumprod([1 + r for r in benchmark_returns])
        
        # Step 6: Plot the results
        plot_cumulative_returns(
            strategy_cumulative_returns,
            benchmark_cumulative_returns,
            title='Cumulative Returns in 2023'
        )
        
        # Step 7: Print summary
        strategy_total_return = strategy_cumulative_returns[-1] - 1
        benchmark_total_return = benchmark_cumulative_returns[-1] - 1
        
        print(f"Total return of PPO Strategy in 2023: {strategy_total_return:.2%}")
        print(f"Total return of Equal-Weighted Portfolio in 2023: {benchmark_total_return:.2%}")

if __name__ == '__main__':
    main()
