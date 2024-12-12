import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, List, Optional
import logging
import time

class DataManager:
    """Simplified Manager for financial data loading and caching"""

    def __init__(self, data_dir: str = 'rl_data'):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        os.makedirs(data_dir, exist_ok=True)

    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_data(self,
                 tickers: List[str],
                 num_assets: int = 10,
                 start_date: str = "2020-01-01",
                 end_date: str = "2022-12-31",
                 force_reload: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load or download financial data.

        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to download.
        num_assets : int
            Number of assets to include.
        start_date : str
            Start date for data collection (YYYY-MM-DD).
        end_date : str
            End date for data collection (YYYY-MM-DD).
        force_reload : bool
            If True, force reload data even if it exists.

        Returns:
        --------
        Tuple containing:
            - X (np.ndarray): Price data of shape (num_days, num_assets).
            - y (np.ndarray): Return data of shape (num_days, num_assets).
            - selected_tickers (List[str]): List of asset tickers.
        """
        try:
            if not force_reload:
                # Try loading from cache first
                cached_data = self._load_from_cache(tickers, num_assets, start_date, end_date)
                if cached_data is not None:
                    return cached_data

            # If not in cache or force_reload, download new data
            self.logger.info("Downloading new data from yfinance...")
            return self._download_and_cache_data(tickers, num_assets, start_date, end_date)

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_from_cache(self, tickers: List[str], num_assets: int, start_date: str, end_date: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Try loading data from cache."""
        # Define filenames with sorted tickers to ensure consistency
        sorted_tickers = sorted(tickers)
        tickers_key = '_'.join(sorted_tickers[:num_assets])
        x_file = f"x_train_{num_assets}_{start_date}_{end_date}.pkl"
        y_file = f"y_train_{num_assets}_{start_date}_{end_date}.pkl"
        tickers_file = f"tickers_{num_assets}_{start_date}_{end_date}.pkl"

        x_path = os.path.join(os.path.join(os.getcwd(), self.data_dir), x_file)
        y_path = os.path.join(os.path.join(os.getcwd(), self.data_dir), y_file)
        tickers_path = os.path.join(os.path.join(os.getcwd(), self.data_dir), tickers_file)

        print("trying to load")

        try:
            with open(x_path, 'rb') as f:
                X = pickle.load(f)
            with open(y_path, 'rb') as f:
                y = pickle.load(f)
            with open(tickers_path, 'rb') as f:
                loaded_tickers = pickle.load(f)
            return X, y, loaded_tickers
        except Exception as e:
            print(e)
            return None

    def _download_and_cache_data(self, tickers: List[str], num_assets: int, start_date: str, end_date: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Download data from yfinance and cache it."""
        selected_tickers = []
        all_prices = []
        all_returns = []

        for ticker in tickers:
            if len(selected_tickers) >= num_assets:
                break  # Stop if desired number of assets is reached
            try:
                self.logger.info(f"Downloading {ticker}...")
                # Download data with threads=False to prevent connection pool issues
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
                
                if data.empty:
                    self.logger.warning(f"Skipping {ticker}: No data downloaded.")
                    continue

                adj_close = data['Adj Close'].dropna()
                if len(adj_close) < 30:
                    self.logger.warning(f"Skipping {ticker}: Insufficient data points ({len(adj_close)}).")
                    continue

                # Calculate returns
                returns = adj_close.pct_change().dropna().values
                prices = adj_close.values[1:]  # Align with returns

                if len(returns) == 0 or len(prices) == 0:
                    self.logger.warning(f"Skipping {ticker}: No returns calculated.")
                    continue

                selected_tickers.append(ticker)
                all_prices.append(prices)
                all_returns.append(returns)
                self.logger.info(f"Added {ticker}: {len(prices)} data points.")

                # Optional: Add a small delay to prevent overwhelming the server
                time.sleep(0.5)

            except Exception as e:
                self.logger.warning(f"Failed to download {ticker}: {str(e)}")

        if len(selected_tickers) < num_assets:
            self.logger.warning(f"Only found {len(selected_tickers)} assets with sufficient data, requested {num_assets}.")

        if not selected_tickers:
            raise ValueError("No valid tickers downloaded.")

        # Find the minimum length to align all assets
        min_length = min(len(prices) for prices in all_prices)

        # Trim all arrays to the minimum length
        X = np.column_stack([prices[-min_length:, :num_assets] for prices in all_prices])  # Shape: (num_days, num_assets)
        y = np.column_stack([returns[-min_length:, :num_assets] for returns in all_returns])  # Shape: (num_days, num_assets)

        # Save to cache
        self._save_to_cache(X, y, selected_tickers[:num_assets], num_assets, start_date, end_date)

        return X, y, selected_tickers

    def _save_to_cache(self, X: np.ndarray, y: np.ndarray, tickers: List[str], 
                      num_assets: int, start_date: str, end_date: str):
        """Save data to cache."""
        sorted_tickers = sorted(tickers)
        tickers_key = '_'.join(sorted_tickers[:num_assets])
        x_file = f"x_train_{len(tickers)}_{start_date}_{end_date}.pkl"
        y_file = f"y_train_{len(tickers)}_{start_date}_{end_date}.pkl"
        tickers_file = f"tickers_{len(tickers)}_{start_date}_{end_date}.pkl"

        x_path = os.path.join(os.path.join(os.getcwd(), self.data_dir), x_file)
        y_path = os.path.join(os.path.join(os.getcwd(), self.data_dir), y_file)
        tickers_path = os.path.join(os.path.join(os.getcwd(), self.data_dir), tickers_file)

        with open(x_path, 'wb') as f:
            pickle.dump(X, f)
        with open(y_path, 'wb') as f:
            pickle.dump(y, f)
        with open(tickers_path, 'wb') as f:
            pickle.dump(tickers, f)

        self.logger.info(f"Saved data to cache: X shape {X.shape}, y shape {y.shape}, tickers {tickers}")

def main():
    """Example usage with sample data loading"""
    # Configure root logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_manager = DataManager()

    # Define periods
    periods = [
        ("2020-01-01", "2022-12-31"),
        ("2022-01-01", "2023-12-31")
    ]

    # Define a list of tickers (for simplicity, using a smaller list)
    # You can adjust this list or read from 'sp500tickers.txt'
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'PG', 'MA',
        'HD', 'BAC', 'CVX', 'KO', 'PFE', 'DIS', 'WMT', 'MRK', 'XOM', 'ORCL'
    ]

    for start_date, end_date in periods:
        try:
            num_assets = 10  # Change as needed
            print(f"\nLoading data for period {start_date} to {end_date} with {num_assets} assets")
            X, y, selected_tickers = data_manager.load_data(
                tickers=tickers,
                num_assets=num_assets,
                start_date=start_date,
                end_date=end_date,
                force_reload=False
            )

            print(f"Data shapes:")
            print(f"X: {X.shape}")  # Expected: (num_days, num_assets)
            print(f"y: {y.shape}")  # Expected: (num_days, num_assets)
            print(f"Number of assets: {len(selected_tickers)}")
            print(f"Sample tickers: {selected_tickers[:5]}")

        except Exception as e:
            print(f"Failed to load data for period {start_date} to {end_date}: {e}")

if __name__ == "__main__":
    main()
