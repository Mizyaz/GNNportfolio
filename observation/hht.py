import numpy as np
import pandas as pd
from PyEMD import EMD
from scipy.signal import hilbert
import yfinance as yf

def preprocess_financial_data(df, column='Close', method='pct_change', fillna=True):
    """
    Preprocesses financial data to a format suitable for HHT analysis.

    @param df: pd.DataFrame
        The input financial data.
    @param column: str
        The column to analyze (e.g., 'Close').
    @param method: str
        Method for processing the data ('pct_change' or 'raw').
    @param fillna: bool
        Whether to fill NaN values.

    @return: np.ndarray
        Preprocessed 1D array of financial data.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    if method == 'pct_change':
        data = df[column].pct_change().values
    elif method == 'raw':
        data = df[column].values
    else:
        raise ValueError("Method must be 'pct_change' or 'raw'.")
    
    if fillna:
        data = np.nan_to_num(data)  # Fill NaNs with 0

    return data.flatten()

def hilbert_huang_transform(data):
    """
    Performs HHT on the given financial data.

    @param data: np.ndarray
        The input 1D financial time series data.

    @return: dict
        Dictionary containing IMFs, instantaneous frequencies, and amplitudes:
        - 'imfs': np.ndarray of shape (num_imfs, len(data))
        - 'instantaneous_freq': np.ndarray of shape (num_imfs, len(data))
        - 'instantaneous_amp': np.ndarray of shape (num_imfs, len(data))
    """
    # Decompose data using EMD
    emd = EMD()
    imfs = emd.emd(data)

    # Apply Hilbert Transform to each IMF
    instantaneous_freq = []
    instantaneous_amp = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        instantaneous_amp.append(np.abs(analytic_signal))  # Amplitude
        instantaneous_freq.append(np.diff(np.unwrap(np.angle(analytic_signal))) / (2.0 * np.pi))

    # Convert lists to arrays
    instantaneous_amp = np.array(instantaneous_amp)
    instantaneous_freq = np.array(instantaneous_freq)

    return {
        "imfs": imfs,
        "instantaneous_freq": instantaneous_freq,
        "instantaneous_amp": instantaneous_amp
    }

def mean_hht(data):
    hht_results = hilbert_huang_transform(data)
    imfs = hht_results['imfs']
    instantaneous_freq = hht_results['instantaneous_freq']
    instantaneous_amp = hht_results['instantaneous_amp']
    return np.mean(imfs), np.mean(instantaneous_freq), np.mean(instantaneous_amp)

def mean_instantaneous_freq(data):
    hht_results = hilbert_huang_transform(data)
    instantaneous_freq = hht_results['instantaneous_freq']
    return np.mean(instantaneous_freq)

def mean_instantaneous_amp(data):
    hht_results = hilbert_huang_transform(data)
    instantaneous_amp = hht_results['instantaneous_amp']
    return np.mean(instantaneous_amp)

def example_hht():

    start_year = 2013
    end_year = 2024

    apple_data_all = yf.download('AAPL', start=f'{start_year}-01-01', end=f'{end_year}-12-31')

    means = []
    means_instantaneous_freq = []
    means_instantaneous_amp = []

    for year in range(start_year, end_year):
        # Example Usage
        for month in range(1, 13):
            # read data
            apple_data = apple_data_all[f'{year}-{month}-01':f'{year}-{month}-28']

            # Preprocess data
            processed_data = preprocess_financial_data(apple_data, column='Close', method='pct_change')

            # Perform HHT
            hht_results = hilbert_huang_transform(processed_data)

            # Extract results
            imfs = hht_results['imfs']
            instantaneous_freq = hht_results['instantaneous_freq']
            instantaneous_amp = hht_results['instantaneous_amp']

            means.append(np.mean(imfs))
            means_instantaneous_freq.append(np.mean(instantaneous_freq))
            means_instantaneous_amp.append(np.mean(instantaneous_amp))

    return means, means_instantaneous_freq, means_instantaneous_amp
