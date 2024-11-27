import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Step 1: Load tickers
with open('sp500tickers.txt', 'r') as f:
    tickers = f.read().splitlines()

# Load or fetch data
try:
    with open('price_df.pkl', 'rb') as f:
        price_df = pickle.load(f)
except FileNotFoundError:
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    start_date = '2021-12-31'
    end_date = '2023-12-31'
    price_df = pd.DataFrame()

    for i, ticker in enumerate(tickers):
        try:
            print(f"Downloading data for {ticker} ({i+1}/{len(tickers)})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
            price_df = pd.concat([price_df, df], axis=1)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    with open('price_df.pkl', 'wb') as f:
        pickle.dump(price_df, f)

# Step 2: Preprocess data
returns_df = price_df.pct_change().dropna()
train_returns = returns_df['2022-01-01':'2022-12-31']
test_returns = returns_df['2023-01-01':'2023-12-31']

# Step 3: Create sequences
def create_sequences(returns, window_size=20, pred_size=5):
    X, y = [], []
    for i in range(len(returns) - window_size - pred_size + 1):
        sequence = returns[i:i+window_size]
        X.append(sequence.reshape(-1, 1))  # Reshape to (window_size, 1)
        y.append(returns[i+window_size:i+window_size+pred_size].sum())
    return np.array(X), np.array(y)

X_train, y_train = [], []
for ticker in train_returns.columns:
    returns = train_returns[ticker].dropna().values
    if len(returns) >= 25:  # window_size + pred_size
        X, y = create_sequences(returns)
        X_train.append(X)
        y_train.append(y)

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

X_test, y_test = [], []
for ticker in test_returns.columns:
    returns = test_returns[ticker].dropna().values
    if len(returns) >= 25:
        X, y = create_sequences(returns)
        X_test.append(X)
        y_test.append(y)

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

print(f"Shapes before standardization: X_train: {X_train.shape}, y_train: {y_train.shape}")

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 4: Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=2, output_size=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x[-1]  # Use the last output token for prediction
        return self.fc(x)

input_size = X_train.shape[2]
model = TransformerModel(input_size)
print(model)

# Step 5: Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 8
batch_size = 4

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / X_train.size(0):.6f}")

# Step 6: Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()

# Step 7: Construct a predictions DataFrame
dates = test_returns.index
predictions_df = pd.DataFrame({
    'Date': np.repeat(dates, X_test.size(0) // len(dates)),
    'PredictedReturn': y_pred.flatten(),
    'ActualReturn': y_test.numpy().flatten()
})

# Step 8: Simulate and evaluate portfolio performance
# (Similar to the LSTM implementation)

print(predictions_df.head())
