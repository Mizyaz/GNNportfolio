# Complete Training Code: Portfolio Prediction using GAT with PyTorch Geometric

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import pickle
import os

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Read tickers from 'sp500tickers.txt'
print("Reading tickers from 'sp500tickers.txt'...")
with open('sp500tickers.txt', 'r') as f:
    tickers = f.read().splitlines()

# Replace '.' with '-' for yfinance compatibility
tickers = [ticker.replace('.', '-') for ticker in tickers]
print(f"Total tickers read: {len(tickers)}")

# Step 2: Download historical data for 2022 and 2023
price_df_path = 'price_df.pkl'
if os.path.exists(price_df_path):
    print("Loading price data from 'price_df.pkl'...")
    with open(price_df_path, 'rb') as f:
        price_df = pickle.load(f)
else:
    print("Price DataFrame not found. Downloading data...")
    start_date = '2021-12-31'
    end_date = '2023-12-31'

    # Initialize an empty DataFrame to hold adjusted close prices
    price_df = pd.DataFrame()

    # Download data for each ticker
    for i, ticker in enumerate(tickers):
        try:
            print(f"Downloading data for {ticker} ({i+1}/{len(tickers)})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"No data for {ticker}. Skipping.")
                continue
            df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
            price_df = pd.concat([price_df, df], axis=1)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    print("Data download complete.")

    with open(price_df_path, 'wb') as f:
        pickle.dump(price_df, f)
    print(f"Price data saved to '{price_df_path}'.")

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

# Step 5: Define a function to create graph data
def create_graph_data(returns, window_size=20):
    """
    Create graph data for each day based on the past window_size days.

    Args:
        returns (pd.DataFrame): DataFrame of returns.
        window_size (int): Number of past days to use as features.

    Returns:
        List[Data]: List of PyTorch Geometric Data objects.
    """
    graph_data_list = []
    dates = returns.index

    # Create features: For each stock, use the past window_size returns as features
    for i in range(window_size, len(returns)):
        date = dates[i]
        X = returns.iloc[i-window_size:i].values  # Shape: (window_size, num_stocks)
        y = returns.iloc[i].values  # Shape: (num_stocks,)

        # Handle missing data by replacing NaNs with zeros
        X = np.nan_to_num(X)

        # **Transpose X to have shape (num_nodes, in_channels)**
        X = X.T  # Shape: (num_nodes, window_size)

        # Convert to torch tensors
        x = torch.tensor(X, dtype=torch.float)  # (480, 20)
        y = torch.tensor(y, dtype=torch.float)  # (480,)

        # Create fully connected edge index
        num_nodes = x.size(0)
        edge_index = []
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst:
                    edge_index.append([src, dst])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.date = date  # Add date attribute for reference
        graph_data_list.append(data)

    return graph_data_list

# Step 6: Prepare training and testing graph data
print("Preparing training graph data...")
os.makedirs('graphs', exist_ok=True)

try:
    with open('graphs/train_graphs.pkl', 'rb') as f:
        train_graphs = pickle.load(f)
except FileNotFoundError:
    train_graphs = create_graph_data(train_returns, window_size=20)
    with open('graphs/train_graphs.pkl', 'wb') as f:
        pickle.dump(train_graphs, f)
print(f"Total training graphs: {len(train_graphs)}")

print("Preparing testing graph data...")
try:
    with open('graphs/test_graphs.pkl', 'rb') as f:
        test_graphs = pickle.load(f)
except FileNotFoundError:
    test_graphs = create_graph_data(test_returns, window_size=20)
    with open('graphs/test_graphs.pkl', 'wb') as f:
        pickle.dump(test_graphs, f)
print(f"Total testing graphs: {len(test_graphs)}")

# Step 7: Create DataLoaders
batch_size = 1  # Since each graph represents a different day
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

# Step 8: Define the GAT Model
class PortfolioGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(PortfolioGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Step 9: Initialize the model, loss function, and optimizer
num_features = 20  # window_size
hidden_channels = 64
output_channels = 1  # Predicting next day return

model = PortfolioGAT(in_channels=num_features, hidden_channels=hidden_channels, out_channels=output_channels)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Step 10: Training Loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze()  # Shape: (num_nodes,)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
    return total_loss / len(loader.dataset)

# Step 11: Evaluation Function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    dates = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index).squeeze()
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_nodes
            predictions.append(out.cpu())
            actuals.append(data.y.cpu())
            dates.append(data.date.cpu() if hasattr(data, 'date') else None)
    avg_loss = total_loss / len(loader.dataset)
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    return avg_loss, predictions, actuals, dates

# Step 12: Training the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

num_epochs = 50
train_losses = []
test_losses = []

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    loss = train(model, train_loader, optimizer, criterion, device)
    train_losses.append(loss)
    if epoch % 5 == 0 or epoch == 1:
        test_loss, _, _, _ = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}/{num_epochs}, Training Loss: {loss:.6f}, Testing Loss: {test_loss:.6f}")

print("Training complete.")

# Plot training and testing loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(5, num_epochs + 1, 5), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Step 13: Make predictions on the test set
print("Making predictions on the test set...")
test_loss, y_pred, y_test, test_dates = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.6f}")

# Step 14: Construct a DataFrame with predictions and actual returns
print("Constructing predictions DataFrame...")
predictions_df = pd.DataFrame(y_pred, columns=['PredictedReturn'])
predictions_df['ActualReturn'] = y_test
# Assign tickers
predictions_df['Ticker'] = tickers[:predictions_df.shape[0]]
# Assign dates (assuming each graph corresponds to a single date and all stocks)
# For simplicity, we'll assign the same date to all stocks per graph
dates_list = []
for graph in test_graphs[:len(test_graphs)].date:
    dates_list.extend([graph] * len(tickers))
predictions_df['Date'] = dates_list[:predictions_df.shape[0]]

print("Sample of Predictions DataFrame:")
print(predictions_df.head())

# Step 15: Simulate the portfolio performance
print("Simulating the portfolio performance...")

# Sort predictions by date
predictions_df.sort_values('Date', inplace=True)

# Initialize an empty DataFrame to hold portfolio weights
unique_dates = predictions_df['Date'].unique()
portfolio = pd.DataFrame(index=unique_dates, columns=tickers).fillna(0)

# For each date, assign weights based on predicted returns
for date in unique_dates:
    daily_predictions = predictions_df[predictions_df['Date'] == date]
    # Rank tickers by predicted return
    daily_predictions = daily_predictions.sort_values('PredictedReturn', ascending=False)
    # Select top N tickers (e.g., top 50)
    top_n = 50
    top_tickers = daily_predictions['Ticker'].head(top_n)
    # Assign equal weights to top tickers
    if len(top_tickers) > 0:
        weights = 1 / len(top_tickers)
        portfolio.loc[date, top_tickers] = weights
    print(f"Date: {date.date()}, Selected {len(top_tickers)} tickers.")

print("Portfolio simulation complete.")

# Step 16: Calculate daily returns of the strategy
print("Calculating daily returns of the strategy...")
# Align the returns DataFrame with the portfolio
aligned_returns = test_returns.loc[portfolio.index]
strategy_returns = (aligned_returns * portfolio.shift(1)).sum(axis=1)
strategy_cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
print("Strategy returns calculated.")

# Step 17: Calculate cumulative returns of the equal-weighted portfolio
print("Calculating equal-weighted portfolio returns...")
equal_weights = pd.DataFrame(1 / len(tickers), index=aligned_returns.index, columns=tickers)
equal_weight_returns = (aligned_returns * equal_weights.shift(1)).sum(axis=1)
equal_weight_cumulative_returns = (1 + equal_weight_returns.fillna(0)).cumprod()
print("Equal-weighted portfolio returns calculated.")

# Step 18: Plot the cumulative returns
print("Plotting cumulative returns...")
plt.figure(figsize=(12, 6))
plt.plot(strategy_cumulative_returns.index, strategy_cumulative_returns.values, label='GAT Strategy')
plt.plot(equal_weight_cumulative_returns.index, equal_weight_cumulative_returns.values, label='Equal-Weighted Portfolio')
plt.title('Cumulative Returns in 2023')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# Step 19: Print the total returns
print(f"Total return of GAT Strategy in 2023: {strategy_cumulative_returns.iloc[-1] - 1:.2%}")
print(f"Total return of Equal-Weighted Portfolio in 2023: {equal_weight_cumulative_returns.iloc[-1] - 1:.2%}")

print("Analysis complete.")
