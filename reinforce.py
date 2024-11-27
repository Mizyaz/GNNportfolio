import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

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

# Load S&P 500 tickers
with open('sp500tickers.txt', 'r') as f:
    tickers = f.read().splitlines()

tickers = [ticker.replace('.', '-') for ticker in tickers]
start_date = '2021-12-31'
end_date = '2023-12-31'

# Download historical data
print("Downloading historical data...")
import pickle
try:
    with open('price_df.pkl', 'rb') as f:
        price_data = pickle.load(f)
except FileNotFoundError:
    price_data = download_data(tickers, start_date, end_date)
    with open('price_df.pkl', 'wb') as f:
        pickle.dump(price_data, f)

# Calculate daily returns
returns_df = price_data.pct_change().dropna()

# Split into training and testing datasets
train_returns = returns_df['2022-01-01':'2022-12-31']
test_returns = returns_df['2023-01-01':'2023-12-31']

# Step 2: Define the custom Gym environment
class TradingEnv(gym.Env):
    def __init__(self, returns_df, window_size=20):
        super(TradingEnv, self).__init__()
        
        self.returns_df = returns_df
        self.tickers = returns_df.columns
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

# Step 3: Train the PPO agent
print("Training the PPO agent...")

# Create the training environment
train_env = lambda : TradingEnv(train_returns)
vec_env = make_vec_env(train_env, n_envs=4)

# Train the agent using PPO
model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, gamma=0.99)
model.learn(total_timesteps=200000)

# Save the model
model.save("ppo_trading_agent")

# Step 4: Test the PPO agent
print("Testing the PPO agent...")

test_env = TradingEnv(test_returns)
obs, _ = test_env.reset()
done = False
cumulative_reward = 0
daily_rewards = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated,_ = test_env.step(action)
    cumulative_reward += reward
    daily_rewards.append(reward)
    test_env.render()

print(f"Cumulative Reward: {cumulative_reward}")

# Step 5: Evaluate performance
print("Evaluating performance...")

# Calculate cumulative returns
strategy_cumulative_returns = np.cumprod([1 + r for r in daily_rewards])

# Benchmark: Equal-weighted portfolio
equal_weights = np.ones(len(test_env.tickers)) / len(test_env.tickers)
benchmark_returns = test_returns.mean(axis=1).values
benchmark_cumulative_returns = np.cumprod([1 + r for r in benchmark_returns])

# Step 6: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(strategy_cumulative_returns, label='PPO Strategy')
plt.plot(benchmark_cumulative_returns, label='Equal-Weighted Portfolio')
plt.title('Cumulative Returns in 2023')
plt.xlabel('Days')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Print summary
strategy_total_return = strategy_cumulative_returns[-1] - 1
benchmark_total_return = benchmark_cumulative_returns[-1] - 1

print(f"Total return of PPO Strategy in 2023: {strategy_total_return:.2%}")
print(f"Total return of Equal-Weighted Portfolio in 2023: {benchmark_total_return:.2%}")
