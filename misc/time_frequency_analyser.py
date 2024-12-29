# time_frequency_analyser.py

import numpy as np
import librosa
from typing import Dict, List, Any
from scipy.stats import entropy
from dataclasses import dataclass

@dataclass
class TFAConfig:
    n_mels: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    window: str = 'hann'
    fmax: float = None  # Maximum frequency for Mel spectrogram
    entropy_bins: int = 10  # Number of bins for entropy calculation

class TimeFrequencyAnalyser:
    """
    Analyzes time-series data using time-frequency representations and computes entropy.
    """
    def __init__(self, config: TFAConfig = TFAConfig()):
        self.config = config
    
    def compute_mel_spectrogram(self, signal: np.ndarray, sr: int = 252) -> np.ndarray:
        """
        Compute the Mel spectrogram of a given signal.
        
        Args:
            signal (np.ndarray): Time-series data.
            sr (int): Sampling rate. For daily data, typical annual trading days are ~252.
        
        Returns:
            np.ndarray: Mel spectrogram in dB.
        """
        S = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            window=self.config.window,
            fmax=self.config.fmax
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    
    def compute_rayleigh_entropy(self, mel_spectrogram: np.ndarray) -> float:
        """
        Compute the Rayleigh entropy of a Mel spectrogram.
        
        Args:
            mel_spectrogram (np.ndarray): Mel spectrogram in dB.
        
        Returns:
            float: Rayleigh entropy value.
        """
        # Flatten the spectrogram and compute histogram
        flattened = mel_spectrogram.flatten()
        hist, bin_edges = np.histogram(flattened, bins=self.config.entropy_bins, density=True)
        hist += 1e-6  # Avoid zero probabilities
        hist /= hist.sum()
        # Compute entropy
        ent = entropy(hist, base=2)
        return ent
    
    def analyze(self, signals: Dict[str, np.ndarray], sr: int = 252) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple time-series signals.
        
        Args:
            signals (Dict[str, np.ndarray]): Dictionary of asset names and their signals.
            sr (int): Sampling rate.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with asset names as keys and their analysis as values.
        """
        analysis_results = {}
        for asset, signal in signals.items():
            mel_spec = self.compute_mel_spectrogram(signal, sr)
            entropy_val = self.compute_rayleigh_entropy(mel_spec)
            analysis_results[asset] = {
                'mel_spectrogram': mel_spec,
                'rayleigh_entropy': entropy_val
            }
        return analysis_results
