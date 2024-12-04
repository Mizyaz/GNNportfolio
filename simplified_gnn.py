# gnn_portfolio.py

import os
import pickle
import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

# Suppress deprecation warnings from PyTorch Geometric
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class BlackLittermanLoss(nn.Module):
    """
    Custom loss function that encourages finding better performing strategies than equal weight
    """
    def __init__(self, returns_df: pd.DataFrame, lam: float = 0.1):
        super(BlackLittermanLoss, self).__init__()
        self.lam = lam
        
        # Calculate equal weight benchmark for reference
        self.num_assets = len(returns_df.columns)
        self.equal_weights = torch.ones(self.num_assets) / self.num_assets

    def forward(self, predictions, targets):
        """
        Loss function that:
        1. Minimizes prediction error (MSE)
        2. Rewards deviation from equal weight when it improves returns
        3. Penalizes deviation that reduces returns
        """
        batch_size = predictions.size(0) // self.num_assets
        
        # Calculate prediction MSE
        mse_loss = F.mse_loss(predictions, targets)
        
        # Get portfolio weights through softmax
        pred_weights = F.softmax(predictions.view(batch_size, self.num_assets), dim=1)
        
        # Calculate portfolio returns
        pred_returns = torch.sum(pred_weights * targets.view(batch_size, self.num_assets), dim=1)
        equal_returns = torch.sum(self.equal_weights.to(predictions.device) * 
                                targets.view(batch_size, self.num_assets), dim=1)
        
        # Calculate return difference from equal weight
        return_diff = pred_returns - equal_returns
        
        # Penalize when returns are worse than equal weight
        # Reward when returns are better
        strategy_loss = -torch.mean(return_diff)
        
        # Combine losses: minimize MSE and maximize return difference
        loss = mse_loss + self.lam * strategy_loss
        
        return loss


class EnhancedPortfolioGAT(nn.Module):
    """
    Enhanced Graph Attention Network for Portfolio Prediction.
    Incorporates batch normalization, skip connections, and a more sophisticated architecture.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: List[int] = [8, 8], dropout: float = 0.6):
        super(EnhancedPortfolioGAT, self).__init__()
        
        # Batch Normalization for input features
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # First GAT layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads[0], dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads[0])
        
        # Second GAT layer
        self.conv2 = GATConv(hidden_channels * heads[0], hidden_channels, heads=heads[1], dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_channels * heads[1])
        
        # Prediction head
        self.pred = nn.Sequential(
            nn.Linear(hidden_channels * heads[1], hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Skip connection
        self.skip = nn.Linear(in_channels, hidden_channels * heads[1])
        
    def forward(self, data):
        """
        Forward pass handling PyG Data objects
        
        Args:
            data: PyG Data object containing x and edge_index
        """
        x, edge_index = data.x, data.edge_index
        
        # Save input for skip connection
        x_in = x
        
        # Main path
        x = self.bn1(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn2(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn3(x)
        
        # Skip connection
        x = x + self.skip(x_in)
        
        # Prediction
        x = self.pred(x)
        return x


class PortfolioPredictor:
    """
    Portfolio Predictor using Graph Attention Networks.
    Handles data processing, model training, evaluation, and portfolio simulation.
    """
    def __init__(self, 
                 tickers_file: str = 'sp500tickers.txt',
                 price_data_file: str = 'price_df.pkl',
                 train_year: int = 2022,
                 test_year: int = 2023,
                 window_size: int = 20,
                 correlation_window: int = 60,
                 correlation_threshold: float = 0.3,
                 top_n: int = 20,  # Number of top predicted stocks to select for the portfolio
                 num_assets: int = 20,  # Number of assets to consider
                 device: str = 'cpu'):
        """
        Initialize the PortfolioPredictor.

        Args:
            tickers_file (str): Path to the file containing stock tickers.
            price_data_file (str): Path to save/load the price DataFrame.
            train_year (int): Year to use for training.
            test_year (int): Year to use for testing.
            window_size (int): Number of past days to use for feature generation.
            correlation_window (int): Window size for rolling correlation to determine edges.
            correlation_threshold (float): Threshold to determine significant correlations for edge creation.
            top_n (int): Number of top predicted stocks to select for the portfolio.
            num_assets (int): Number of top assets to select based on average returns.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.tickers_file = tickers_file
        self.price_data_file = price_data_file
        self.train_year = train_year
        self.test_year = test_year
        self.window_size = window_size
        self.correlation_window = correlation_window
        self.correlation_threshold = correlation_threshold
        self.top_n = top_n
        self.num_assets = num_assets
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize placeholders
        self.tickers = []
        self.price_df = None
        self.returns_df = None
        self.train_returns = None
        self.test_returns = None
        self.train_features = None
        self.test_features = None
        self.train_graphs = []
        self.test_graphs = []
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping = None
        self.scaler = None

    def load_tickers(self):
        """Load tickers from the specified file and select top N assets based on average returns."""
        print(f"Reading tickers from '{self.tickers_file}'...")
        with open(self.tickers_file, 'r') as f:
            all_tickers = f.read().splitlines()
        # Replace '.' with '-' for yfinance compatibility
        all_tickers = [ticker.replace('.', '-') for ticker in all_tickers]
        print(f"Total tickers read: {len(all_tickers)}")
        
        # Placeholder for selected tickers
        selected_tickers = []
        
        # Select top N tickers based on average returns (excluding NaNs)
        if self.train_returns is not None and not self.train_returns.empty:
            average_returns = self.train_returns.mean().sort_values(ascending=False)
            selected_tickers = average_returns.head(self.num_assets).index.tolist()
            print(f"Selected top {self.num_assets} tickers based on average returns.")
        else:
            # If train_returns not available yet, select first N tickers
            selected_tickers = all_tickers[:self.num_assets]
            print(f"No training returns available yet. Selecting first {self.num_assets} tickers.")
        
        self.tickers = selected_tickers[:self.num_assets]
        print(f"Selected tickers: {self.tickers}")

    def download_price_data(self):
        """Download historical price data or load from pickle if available."""
        if os.path.exists(self.price_data_file):
            print(f"Loading price data from '{self.price_data_file}'...")
            with open(self.price_data_file, 'rb') as f:
                self.price_df = pickle.load(f)
        else:
            print("Price DataFrame not found. Downloading data...")
            start_date = f'{self.train_year - 1}-12-31'
            end_date = f'{self.test_year}-12-31'

            # Initialize an empty DataFrame to hold adjusted close prices
            self.price_df = pd.DataFrame()

            # Download data for each ticker
            for i, ticker in enumerate(self.tickers):
                try:
                    print(f"Downloading data for {ticker} ({i+1}/{len(self.tickers)})...")
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if df.empty:
                        print(f"No data for {ticker}. Skipping.")
                        continue
                    df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
                    self.price_df = pd.concat([self.price_df, df], axis=1)
                except Exception as e:
                    print(f"Error downloading data for {ticker}: {e}")

            print("Data download complete.")

            with open(self.price_data_file, 'wb') as f:
                pickle.dump(self.price_df, f)
            print(f"Price data saved to '{self.price_data_file}'.")

    def calculate_returns(self):
        """Calculate daily returns from price data."""
        print("Calculating daily returns...")
        self.returns_df = self.price_df.pct_change().dropna()
        print("Daily returns calculated.")

    def split_data(self):
        """Split returns data into training and testing sets based on specified years."""
        print("Splitting data into training and testing sets...")
        train_start = f'{self.train_year}-01-01'
        train_end = f'{self.train_year}-12-31'
        test_start = f'{self.test_year}-01-01'
        test_end = f'{self.test_year}-12-31'
        self.train_returns = self.returns_df.loc[train_start:train_end]
        self.test_returns = self.returns_df.loc[test_start:test_end]
        print(f"Training data range: {self.train_returns.index.min()} to {self.train_returns.index.max()}")
        print(f"Testing data range: {self.test_returns.index.min()} to {self.test_returns.index.max()}")

    def generate_features(self):
        """
        Generate enhanced features for the returns data.
        This includes multiple rolling statistics and technical indicators.
        """
        print("Generating enhanced features...")
        # Example: Using rolling means, stds, and moving averages
        feature_dfs_train = []
        feature_dfs_test = []

        # Rolling means
        for window in [5, 10, 20]:
            feature_dfs_train.append(self.train_returns.rolling(window).mean().rename(columns=lambda x: f'{x}_mean_{window}'))
            feature_dfs_test.append(self.test_returns.rolling(window).mean().rename(columns=lambda x: f'{x}_mean_{window}'))

        # Rolling standard deviations
        for window in [10, 20]:
            feature_dfs_train.append(self.train_returns.rolling(window).std().rename(columns=lambda x: f'{x}_std_{window}'))
            feature_dfs_test.append(self.test_returns.rolling(window).std().rename(columns=lambda x: f'{x}_std_{window}'))

        # Rolling skewness
        feature_dfs_train.append(self.train_returns.rolling(20).skew().rename(columns=lambda x: f'{x}_skew_20'))
        feature_dfs_test.append(self.test_returns.rolling(20).skew().rename(columns=lambda x: f'{x}_skew_20'))

        # Concatenate all features
        self.train_features = pd.concat([self.train_returns] + feature_dfs_train, axis=1).dropna()
        self.test_features = pd.concat([self.test_returns] + feature_dfs_test, axis=1).dropna()

        # Fill any remaining NaNs
        self.train_features = self.train_features.fillna(0)
        self.test_features = self.test_features.fillna(0)

        print("Enhanced features generated.")

    def create_sophisticated_edge_index(self, returns: pd.DataFrame) -> torch.Tensor:
        """
        Create a more sophisticated edge index based on partial correlations and mutual information.

        Args:
            returns (pd.DataFrame): DataFrame of returns.

        Returns:
            torch.Tensor: Edge index tensor.
        """
        print("Creating sophisticated edge index based on partial correlations and mutual information...")
        num_nodes = len(self.tickers)
        edges = set()

        # Compute partial correlations using Ledoit-Wolf shrinkage estimator
        lw = LedoitWolf()
        covariance = lw.fit(returns).covariance_
        try:
            partial_corr_matrix = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            print("Covariance matrix is singular. Using pseudo-inverse.")
            partial_corr_matrix = np.linalg.pinv(covariance)
        diag = np.sqrt(np.diag(partial_corr_matrix))
        # Avoid division by zero
        diag[diag == 0] = 1e-10
        partial_corr_matrix /= diag[:, None]
        partial_corr_matrix /= diag[None, :]
        np.fill_diagonal(partial_corr_matrix, 0)  # Remove self-correlation

        # Threshold for partial correlation
        threshold_pc = self.correlation_threshold

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if abs(partial_corr_matrix[i, j]) > threshold_pc:
                    edges.add((i, j))
                    edges.add((j, i))  # Ensure bidirectional edges

        # Compute mutual information and add edges based on MI threshold
        mi_threshold = 0.05  # Example threshold, can be tuned
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                mi = mutual_info_regression(returns.iloc[:, i].values.reshape(-1, 1),
                                           returns.iloc[:, j].values)[0]
                if mi > mi_threshold:
                    edges.add((i, j))
                    edges.add((j, i))

        if not edges:
            # Fallback to fully connected if no edges meet the threshold
            print("No edges met the thresholds. Falling back to fully connected graph.")
            edge_index = []
            for src in range(num_nodes):
                for dst in range(num_nodes):
                    if src != dst:
                        edge_index.append([src, dst])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            print(f"Total edges created (fully connected): {edge_index.size(1)}")
            return edge_index

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        print(f"Total edges created (sophisticated): {edge_index.size(1)}")
        return edge_index

    def create_graph_data(self, returns: pd.DataFrame, features: pd.DataFrame, edges: torch.Tensor) -> List[Data]:
        """
        Create graph data with proper input and prediction windows
        
        Args:
            returns: DataFrame of returns
            features: DataFrame of enhanced features
            edges: Edge index tensor
        """
        print("Creating graph data...")
        graph_data_list = []
        
        input_window = self.window_size  # e.g., 20 days
        pred_window = 5  # 5-day prediction window
        
        # Scale features
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(features.iloc[:input_window])
        
        features_scaled = pd.DataFrame(
            self.scaler.transform(features), 
            index=features.index, 
            columns=features.columns
        )
        
        # Create graphs with rolling windows
        for i in range(input_window, len(features_scaled) - pred_window):
            # Input features from past window_size days
            X = features_scaled.iloc[i - input_window:i].values
            
            # Target: next pred_window days returns
            future_returns = returns.iloc[i:i + pred_window]
            y = future_returns.mean().values  # Average return over prediction window
            
            # Convert to tensors
            x = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            
            # Create graph with date reference
            data = Data(x=x, edge_index=edges, y=y)
            data.date = features.index[i]
            graph_data_list.append(data)
        
        print(f"Created {len(graph_data_list)} graphs")
        return graph_data_list

    def prepare_graph_data(self):
        """Prepare and save/load graph data for training and testing."""
        # Define pickle file paths
        train_graph_pickle = f'train_graphs_{self.num_assets}.pkl'
        test_graph_pickle = f'test_graphs_{self.num_assets}.pkl'
        edges_pickle = f'edges_{self.num_assets}.pkl'

        if os.path.exists(train_graph_pickle) and os.path.exists(test_graph_pickle) and os.path.exists(edges_pickle):
            print("Loading graph data from pickle files...")
            with open(train_graph_pickle, 'rb') as f:
                self.train_graphs = pickle.load(f)
            with open(test_graph_pickle, 'rb') as f:
                self.test_graphs = pickle.load(f)
            with open(edges_pickle, 'rb') as f:
                edges = pickle.load(f)
            print("Graph data loaded from pickle files.")
        else:
            # Select top assets
            self.select_top_assets()

            # Generate enhanced features
            self.generate_features()

            # Create edges based on training returns using a sophisticated method
            edges = self.create_sophisticated_edge_index(self.train_returns)

            # Create graph data
            self.train_graphs = self.create_graph_data(self.train_returns, self.train_features, edges)
            self.test_graphs = self.create_graph_data(self.test_returns, self.test_features, edges)

            # Save graph data
            with open(train_graph_pickle, 'wb') as f:
                pickle.dump(self.train_graphs, f)
            with open(test_graph_pickle, 'wb') as f:
                pickle.dump(self.test_graphs, f)
            with open(edges_pickle, 'wb') as f:
                pickle.dump(edges, f)
            print("Graph data saved to pickle files.")

    def select_top_assets(self):
        """Select top N assets based on average returns from the training set."""
        print("Selecting top assets based on average returns...")
        average_returns = self.train_returns.mean().sort_values(ascending=False)
        selected_tickers = average_returns.head(self.num_assets).index.tolist()
        self.tickers = selected_tickers[:self.num_assets]
        print(f"Selected {self.num_assets} tickers: {self.tickers}")

        # Filter price and returns data to include only selected tickers
        self.price_df = self.price_df[self.tickers]
        self.train_returns = self.train_returns[self.tickers]
        self.test_returns = self.test_returns[self.tickers]

        # Generate features again since tickers have changed
        self.generate_features()

        # Filter features accordingly
        feature_columns = self.tickers.copy()
        for window in [5, 10, 20]:
            feature_columns += [f'{ticker}_mean_{window}' for ticker in self.tickers]
        for window in [10, 20]:
            feature_columns += [f'{ticker}_std_{window}' for ticker in self.tickers]
        feature_columns += [f'{ticker}_skew_20' for ticker in self.tickers]

        # Check if all feature columns exist
        missing_columns = [col for col in feature_columns if col not in self.train_features.columns]
        if missing_columns:
            print(f"Missing feature columns: {missing_columns}")
            raise KeyError(f"{missing_columns} not in index")

        self.train_features = self.train_features[feature_columns].dropna()
        self.test_features = self.test_features[feature_columns].dropna()

        print(f"Filtered price and returns data for selected tickers.")

    def create_datasets(self):
        """Create PyTorch Geometric datasets from graphs with validation split."""
        from torch_geometric.data import Data, Dataset
        
        class GraphDataset(Dataset):
            def __init__(self, graphs):
                super().__init__()
                self.graphs = graphs
                
            def len(self):
                return len(self.graphs)
                
            def get(self, idx):
                return self.graphs[idx]
        
        # Split training data into train and validation sets (80-20 split)
        train_size = int(0.8 * len(self.train_graphs))
        self.val_graphs = self.train_graphs[train_size:]
        self.train_graphs = self.train_graphs[:train_size]
        
        print(f"Training graphs: {len(self.train_graphs)}")
        print(f"Validation graphs: {len(self.val_graphs)}")
        
        # Create train and validation datasets
        self.train_dataset = GraphDataset(self.train_graphs)
        self.val_dataset = GraphDataset(self.val_graphs)

    def initialize_model(self):
        """Initialize the GAT model, loss function, optimizer, scheduler, and early stopping."""
        print("Initializing the model...")
        
        # Create datasets first
        self.create_datasets()
        
        in_channels = self.train_graphs[0].x.shape[1]  # Number of features
        hidden_channels = 64
        out_channels = self.num_assets  # Predicting weights for each asset

        self.model = EnhancedPortfolioGAT(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            heads=[8, 8],
            dropout=0.6
        ).to(self.device)
        print(self.model)

        # Initialize Black-Litterman inspired loss with return-based weights
        self.criterion = BlackLittermanLoss(
            returns_df=self.train_returns,
            lam=0.1
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            verbose=True
        )
        self.early_stopping = EarlyStopping(patience=20, min_delta=1e-4)

    def calculate_strategy_returns(self, predictions: torch.Tensor, returns: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate strategy returns and equal weight returns for a batch of predictions.
        
        Args:
            predictions: Model predictions (batch_size, num_assets)
            returns: Actual returns (batch_size, num_assets)
        
        Returns:
            tuple: (strategy_return, equal_weight_return)
        """
        # Convert to numpy for calculations
        pred_np = predictions.detach().cpu().numpy()
        returns_np = returns.detach().cpu().numpy()
        
        # Corrected the softmax dimension and ensured numerical stability
        weights = F.softmax(torch.tensor(pred_np), dim=1).numpy()
        
        # Ensure weights are properly normalized
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Calculate strategy returns
        strategy_return = np.sum(weights * returns_np, axis=1).mean()
        
        # Calculate equal weight returns
        equal_weights = np.ones_like(weights) / weights.shape[1]
        equal_weight_return = np.sum(equal_weights * returns_np, axis=1).mean()
        
        return strategy_return, equal_weight_return

    def train_model(self, num_epochs=100, batch_size=16):
        """Train the GNN model with improved monitoring and strategy performance tracking"""
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        # Initialize return tracking
        strategy_returns = []
        equal_weight_returns = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            epoch_strategy_return = 0.0
            epoch_equal_weight_return = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                
                # {{ edit_7 }} Corrected return calculations to prevent compounding within epoch
                strategy_ret, equal_ret = self.calculate_strategy_returns(out, batch.y)
                epoch_strategy_return += strategy_ret
                epoch_equal_weight_return += equal_ret
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_count % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count
            
            # Track returns
            strategy_returns.append(epoch_strategy_return)
            equal_weight_returns.append(epoch_equal_weight_return)
            
            # Print epoch statistics
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Average Loss: {avg_epoch_loss:.4f}")
            print(f"Strategy Return: {(strategy_ret)*100:.2f}%")
            print(f"Equal Weight Return: {(equal_ret)*100:.2f}%")
            cumulative_strategy_return = np.prod([1 + r for r in strategy_returns]) - 1
            cumulative_equal_weight_return = np.prod([1 + r for r in equal_weight_returns]) - 1
            print(f"Cumulative Strategy Return: {(cumulative_strategy_return)*100:.2f}%")
            print(f"Cumulative Equal Weight Return: {(cumulative_equal_weight_return)*100:.2f}%\n")
            
            # Learning rate scheduling
            self.scheduler.step(avg_epoch_loss)
            
            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Validate model periodically
            if epoch % 5 == 0:
                self.validate_model()
        
        # {{ edit_8 }} Updated plotting to reflect corrected return tracking
        self.plot_training_returns(strategy_returns, equal_weight_returns)
        
        # Load best model after training
        self.model.load_state_dict(torch.load('best_model.pth'))

    def plot_training_returns(self, strategy_returns: List[float], equal_weight_returns: List[float]):
        """Plot cumulative returns during training."""
        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative returns
        strategy_cum_returns = np.cumprod(strategy_returns)
        equal_weight_cum_returns = np.cumprod(equal_weight_returns)
        
        plt.plot(strategy_cum_returns, label='Strategy')
        plt.plot(equal_weight_cum_returns, label='Equal Weight')
        plt.title('Training Performance: Strategy vs Equal Weight')
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_returns.png')
        plt.close()

    def validate_model(self):
        """
        Validate the model on the validation dataset
        """
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=32)
        total_val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                total_val_loss += loss.item()
                batch_count += 1
        
        avg_val_loss = total_val_loss / batch_count
        print(f"Validation Loss: {avg_val_loss:.4f}")
        self.model.train()

    def evaluate_model(self, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the model on the test set and return predictions and actuals."""
        print("Evaluating the model on the test set...")
        test_loader = DataLoader(self.test_graphs, batch_size=batch_size, shuffle=False)
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                # Pass the entire batch object to the model
                out = self.model(batch)
                predictions.append(out.cpu().numpy())
                actuals.append(batch.y.cpu().numpy())
        
        y_pred = np.concatenate(predictions).flatten()
        y_test = np.concatenate(actuals).flatten()
        print("Evaluation complete.")
        return y_pred, y_test

    def simulate_portfolio(self, y_pred: np.ndarray, y_test: np.ndarray, print_selection: bool = True) -> pd.DataFrame:
        """
        Simulate portfolio performance based on model predictions.
        
        Args:
            y_pred (np.ndarray): Predicted returns (num_graphs * num_assets,)
            y_test (np.ndarray): Actual returns (num_graphs * num_assets,)
            print_selection (bool): Whether to print selected tickers.
        
        Returns:
            pd.DataFrame: DataFrame containing portfolio weights and returns.
        """
        num_graphs = len(self.test_graphs)
        tickers = self.tickers
        num_tickers = len(tickers)

        # {{ edit_3 }} Corrected the DataFrame construction for predictions
        predictions_df = pd.DataFrame({
            'Graph': np.repeat(range(num_graphs), num_tickers),
            'Ticker': np.tile(tickers, num_graphs),
            'PredictedReturn': y_pred.flatten(),  # Ensure correct shape
            'ActualReturn': y_test.flatten()
        })

        # Initialize portfolio weights DataFrame
        portfolio = pd.DataFrame(index=self.test_returns.index, columns=tickers, dtype=np.float32).fillna(0)
        
        # Initialize equal weight portfolio (constant through time)
        equal_weight_portfolio = pd.DataFrame(1.0/num_tickers, 
                                            index=self.test_returns.index, 
                                            columns=tickers)
        
        # Calculate strategy weights for each day
        for graph_idx, date in enumerate(self.test_returns.index):
            daily_predictions = predictions_df[predictions_df['Graph'] == graph_idx]
            if daily_predictions.empty:
                continue
            daily_predictions = daily_predictions.sort_values('PredictedReturn', ascending=False)
            top_tickers = daily_predictions['Ticker'].head(self.top_n)
            if not top_tickers.empty:
                weight = 1.0 / len(top_tickers)
                portfolio.loc[date, top_tickers] = weight
            if print_selection:
                print(f"Date: {date.date()}, Selected {len(top_tickers)} tickers: {list(top_tickers)}")
        
        # {{ edit_4 }} Align portfolio shifts correctly to prevent look-ahead bias
        strategy_returns = (self.test_returns * portfolio.shift(1)).sum(axis=1)
        strategy_cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    
        # Calculate equal-weighted portfolio returns (constant weights)
        equal_weight_returns = (self.test_returns * equal_weight_portfolio.shift(1)).sum(axis=1)
        equal_weight_cumulative_returns = (1 + equal_weight_returns.fillna(0)).cumprod()
    
        # Compile results
        portfolio_returns = pd.DataFrame({
            'GAT Strategy': strategy_cumulative_returns,
            'Equal-Weighted Portfolio': equal_weight_cumulative_returns
        })
    
        return portfolio_returns

    def calculate_performance_metrics(self, portfolio_returns: pd.DataFrame):
        """Calculate and print performance metrics like Sharpe Ratio and Maximum Drawdown."""
        print("Calculating performance metrics...")

        # Daily returns
        strategy_daily = portfolio_returns['GAT Strategy'].pct_change().dropna()
        equal_daily = portfolio_returns['Equal-Weighted Portfolio'].pct_change().dropna()

        # Sharpe Ratio
        strategy_sharpe = self.calculate_sharpe_ratio(strategy_daily)
        equal_sharpe = self.calculate_sharpe_ratio(equal_daily)

        # Maximum Drawdown
        strategy_mdd = self.calculate_max_drawdown(portfolio_returns['GAT Strategy'])
        equal_mdd = self.calculate_max_drawdown(portfolio_returns['Equal-Weighted Portfolio'])

        print(f"GAT Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"Equal-Weighted Portfolio Sharpe Ratio: {equal_sharpe:.2f}")
        print(f"GAT Strategy Maximum Drawdown: {strategy_mdd:.2%}")
        print(f"Equal-Weighted Portfolio Maximum Drawdown: {equal_mdd:.2%}")

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe Ratio.

        Args:
            returns (pd.Series): Daily returns.
            risk_free_rate (float): Annual risk-free rate.

        Returns:
            float: Sharpe Ratio.
        """
        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        return sharpe

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """
        Calculate the Maximum Drawdown.

        Args:
            cumulative_returns (pd.Series): Cumulative returns.

        Returns:
            float: Maximum Drawdown.
        """
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def plot_cumulative_returns(self, portfolio_returns: pd.DataFrame):
        """Plot cumulative returns for strategy and benchmark."""
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_returns.index, portfolio_returns['GAT Strategy'], 
                 label='GAT Strategy', color='blue')
        plt.plot(portfolio_returns.index, portfolio_returns['Equal-Weighted Portfolio'], 
                 label='Equal-Weighted Portfolio', color='orange')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_returns.png')
        plt.close()

    def run(self):
        """Execute the entire workflow."""
        self.load_tickers()
        self.download_price_data()
        self.calculate_returns()
        self.split_data()
        self.select_top_assets()
        self.prepare_graph_data()
        self.initialize_model()
        self.train_model(num_epochs=100, batch_size=16)
        y_pred, y_test = self.evaluate_model(batch_size=16)
        portfolio_returns = self.simulate_portfolio(y_pred, y_test)
        self.plot_cumulative_returns(portfolio_returns)
        self.calculate_performance_metrics(portfolio_returns)
        print("Workflow complete.")

    def __call__(self):
        """Allow the class instance to be called directly."""
        self.run()


if __name__ == "__main__":
    predictor = PortfolioPredictor(
        tickers_file='sp500tickers.txt',
        price_data_file='price_df.pkl',
        train_year=2022,
        test_year=2023,
        window_size=20,
        correlation_window=60,
        correlation_threshold=0.3,
        top_n=20,  # Number of top predicted stocks to select for the portfolio
        num_assets=20,  # Number of assets to consider
        device='cuda'  # Change to 'cpu' if CUDA is not available
    )
    predictor()
