import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, List, Optional, Dict
from datetime import datetime
import hashlib
import logging

class DataManager:
    """Manager for financial data loading and caching"""
    
    def __init__(self, cache_dir: str = 'data_cache'):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        os.makedirs(cache_dir, exist_ok=True)
        
    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _generate_cache_key(self, params: Dict) -> str:
        """Generate unique cache key based on parameters"""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str, data_type: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{data_type}_{key}.pkl")
    
    def load_data(self,
                 num_assets: int = 10,
                 start_date: str = "2020-01-01",
                 end_date: str = "2022-12-31",
                 force_reload: bool = False,
                 include_features: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
            If True, force reload data even if cached
        include_features : bool
            If True, include additional features
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, List[str]]
            prices, returns, and ticker list
        """
        params = {
            'num_assets': num_assets,
            'start_date': start_date,
            'end_date': end_date,
            'include_features': include_features
        }
        cache_key = self._generate_cache_key(params)
        
        if not force_reload:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                self.logger.info("Using cached data")
                return cached_data
        
        self.logger.info(f"Downloading data for {num_assets} assets...")
        
        # Load tickers
        tickers = self._load_tickers(num_assets)
        
        # Download data
        data = self._download_data(tickers, start_date, end_date)
        
        # Process data
        prices, returns = self._process_data(data)
        
        # Add features if requested
        if include_features:
            features = self._calculate_features(prices, returns)
            processed_data = (prices, returns, features, tickers)
        else:
            processed_data = (prices, returns, tickers)
        
        # Cache the results
        self._save_to_cache(cache_key, processed_data)
        
        return processed_data
    
    def _load_from_cache(self, key: str) -> Optional[Tuple]:
        """Load data from cache"""
        try:
            cache_path = self._get_cache_path(key, 'data')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, key: str, data: Tuple):
        """Save data to cache"""
        try:
            cache_path = self._get_cache_path(key, 'data')
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
    
    def _load_tickers(self, num_assets: int) -> List[str]:
        """Load SP500 tickers"""
        with open('sp500tickers.txt', 'r') as f:
            return [line.strip() for line in f.readlines()][:num_assets]
    
    def _download_data(self, 
                      tickers: List[str], 
                      start_date: str, 
                      end_date: str) -> pd.DataFrame:
        """Download data for given tickers"""
        all_data = {}
        
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock_data.empty:
                    all_data[ticker] = stock_data['Adj Close']
            except Exception as e:
                self.logger.warning(f"Error downloading {ticker}: {e}")
        
        return pd.DataFrame(all_data)
    
    def _process_data(self, 
                     data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw data into prices and returns"""
        prices = data.values
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        return prices, returns
    
    def _calculate_features(self, 
                          prices: np.ndarray, 
                          returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate additional features"""
        features = {}
        
        # Volatility
        features['volatility'] = self._calculate_volatility(returns)
        
        # Moving averages
        features['ma_50'] = self._calculate_ma(prices, 50)
        features['ma_200'] = self._calculate_ma(prices, 200)
        
        # Momentum
        features['momentum'] = self._calculate_momentum(prices)
        
        return features
    
    def _calculate_volatility(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate rolling volatility"""
        return np.array([np.std(returns[max(0, i-window):i], axis=0) 
                        for i in range(len(returns))])
    
    def _calculate_ma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        return np.array([np.mean(prices[max(0, i-window):i], axis=0) 
                        for i in range(len(prices))])
    
    def _calculate_momentum(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate price momentum"""
        return np.array([(prices[i] - prices[max(0, i-window)]) / prices[max(0, i-window)]
                        for i in range(len(prices))])

def main():
    """Example usage"""
    data_manager = DataManager()
    
    # Load data
    prices, returns, tickers = data_manager.load_data(
        num_assets=10,
        start_date="2020-01-01",
        end_date="2022-12-31",
        include_features=False
    )
    
    print("\nData Loading Results:")
    print(f"Number of assets: {len(tickers)}")
    print(f"Price data shape: {prices.shape}")
    print(f"Return data shape: {returns.shape}")
    print("\nSample tickers:", tickers[:5])

if __name__ == "__main__":
    main() 