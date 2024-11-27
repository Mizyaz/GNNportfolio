import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import torch
import torch.nn as nn
import torch.optim as optim




# Step 1: Read tickers from 'sp500tickers.txt'

print("Reading tickers from 'sp500tickers.txt'...")
with open('sp500tickers.txt', 'r') as f:
    tickers = f.read().splitlines()

try:
    import pickle
    with open('price_df.pkl', 'rb') as f:
        price_df = pickle.load(f)
except FileNotFoundError:
    print("Price DataFrame not found. Downloading data...")


    # Remove tickers that may cause issues
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    print(f"Total tickers read: {len(tickers)}")

    # Step 2: Download historical data for 2022 and 2023
    print("Downloading historical data for 2022 and 2023...")
    start_date = '2021-12-31'
    end_date = '2023-12-31'

    # Initialize an empty DataFrame to hold adjusted close prices
    price_df = pd.DataFrame()

    # Download data for each ticker
    for i, ticker in enumerate(tickers):
        try:
            print(f"Downloading data for {ticker} ({i+1}/{len(tickers)})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
            price_df = pd.concat([price_df, df], axis=1)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    print("Data download complete.")

    import pickle

    with open('price_df.pkl', 'wb') as f:
        pickle.dump(price_df, f)

# Step 3: Calculate daily returns
print("Calculating daily returns...")
returns_df = price_df.pct_change().dropna()
print("Daily returns calculated.")

# Step 4: Split data into training and testing sets
print("Splitting data into training and testing sets...")
train_returns = returns_df['2022-01-01':'2022-12-31']
test_returns = returns_df['2023-01-01':'2023-12-31']
print(f"Training data range: {train_returns.index.min()} to {train_returns.index.max()}")
print(f"Testing data range: {test_returns.index.min()} to {test_returns.index.max()}")

# Step 5: Define a function to create sequences
def create_sequences_with_info(returns, dates, ticker, window_size=20, pred_size=5):
    X = []
    y = []
    sample_dates = []
    sample_tickers = []
    for i in range(len(returns) - window_size - pred_size + 1):
        X_seq = returns[i:i+window_size]
        y_seq = returns[i+window_size:i+window_size+pred_size]
        X.append(X_seq)
        y.append(y_seq.sum())
        sample_dates.append(dates[i+window_size+pred_size-1])
        sample_tickers.append(ticker)
    return np.array(X), np.array(y), sample_dates, sample_tickers

# Step 6: Prepare training data
print("Preparing training data...")
X_train_list = []
y_train_list = []
dates_train_list = []
tickers_train_list = []

for ticker in tickers:
    returns = train_returns[ticker].dropna()
    if len(returns) < 25:
        print(f"Skipping {ticker} due to insufficient data.")
        continue
    X_ticker, y_ticker, sample_dates, sample_tickers = create_sequences_with_info(
        returns.values, returns.index, ticker)
    X_train_list.append(X_ticker)
    y_train_list.append(y_ticker)
    dates_train_list.extend(sample_dates)
    tickers_train_list.extend(sample_tickers)
    print(f"Processed training data for {ticker}.")

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)

with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

print(f"Total training samples: {X_train.shape[0]}")

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

print("Training data preparation complete.")

# Step 7: Prepare testing data
print("Preparing testing data...")
X_test_list = []
y_test_list = []
dates_test_list = []
tickers_test_list = []

for ticker in tickers:
    returns = test_returns[ticker].dropna()
    if len(returns) < 25:
        print(f"Skipping {ticker} due to insufficient data.")
        continue
    X_ticker, y_ticker, sample_dates, sample_tickers = create_sequences_with_info(
        returns.values, returns.index, ticker)
    X_test_list.append(X_ticker)
    y_test_list.append(y_ticker)
    dates_test_list.extend(sample_dates)
    tickers_test_list.extend(sample_tickers)
    print(f"Processed testing data for {ticker}.")

X_test = np.concatenate(X_test_list)
y_test = np.concatenate(y_test_list)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print(f"Total testing samples: {X_test.shape[0]}")
print("Testing data preparation complete.")

# Step 8: Build and train the LSTM model
print("Building the LSTM model...")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

model = LSTMModel()
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 9: Train the model
print("Training the model...")
num_epochs = 10
batch_size = 32

train_losses = []

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0.0
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
    
    avg_loss = epoch_loss / X_train.size(0)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

print("Model training complete.")

# Step 10: Make predictions on the test set
print("Making predictions on the test set...")
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

print("Predictions complete.")

# Step 11: Construct a DataFrame with predictions and actual returns
print("Constructing predictions DataFrame...")
predictions_df = pd.DataFrame({
    'Date': dates_test_list,
    'Ticker': tickers_test_list,
    'PredictedReturn': y_pred.numpy().flatten(),
    'ActualReturn': y_test.numpy().flatten()
})

print("Predictions DataFrame constructed.")
print(predictions_df.head())

# Step 12: Simulate the portfolio performance
print("Simulating the portfolio performance...")

# Sort predictions by date
predictions_df.sort_values('Date', inplace=True)

# Initialize an empty DataFrame to hold portfolio weights
unique_dates = predictions_df['Date'].unique()
portfolio = pd.DataFrame(index=unique_dates, columns=tickers).fillna(0)

print(f"Total trading days: {len(unique_dates)}")

# For each date, assign weights based on predicted returns
for date in unique_dates:
    daily_predictions = predictions_df[predictions_df['Date'] == date]
    # Rank tickers by predicted return
    daily_predictions = daily_predictions.sort_values('PredictedReturn', ascending=False)
    # Select top 50 tickers
    top_tickers = daily_predictions['Ticker'].head(50)
    # Assign equal weights to top tickers
    weights = 1 / len(top_tickers)
    portfolio.loc[date, top_tickers] = weights
    print(f"Date: {date.date()}, Selected {len(top_tickers)} tickers.")

print("Portfolio simulation complete.")

# Step 13: Calculate daily returns of the strategy
print("Calculating daily returns of the strategy...")
# Align the returns DataFrame with the portfolio
strategy_returns = (test_returns.loc[portfolio.index] * portfolio.shift()).sum(axis=1)

# Calculate cumulative returns of the strategy
strategy_cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()

print("Strategy returns calculated.")

# Step 14: Calculate cumulative returns of the equal-weighted portfolio
print("Calculating equal-weighted portfolio returns...")
equal_weights = pd.DataFrame(1 / len(tickers), index=portfolio.index, columns=tickers)
equal_weight_returns = (test_returns.loc[equal_weights.index] * equal_weights.shift()).sum(axis=1)
equal_weight_cumulative_returns = (1 + equal_weight_returns.fillna(0)).cumprod()

print("Equal-weighted portfolio returns calculated.")

# Step 15: Plot the cumulative returns
print("Plotting cumulative returns...")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(strategy_cumulative_returns.index, strategy_cumulative_returns.values, label='LSTM Strategy')
plt.plot(equal_weight_cumulative_returns.index, equal_weight_cumulative_returns.values, label='Equal-Weighted Portfolio')
plt.title('Cumulative Returns in 2023')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# Step 16: Print the total returns
print(f"Total return of LSTM Strategy in 2023: {strategy_cumulative_returns[-1] - 1:.2%}")
print(f"Total return of Equal-Weighted Portfolio in 2023: {equal_weight_cumulative_returns[-1] - 1:.2%}")

print("Analysis complete.")
