import numpy as np
from scipy import signal, interpolate
import warnings

def extract_rr_intervals(heart_rate_signal, sampling_rate):
    """
    Extract RR intervals from the heart rate signal using peak detection.
    
    Args:
        heart_rate_signal: Array of heart rate values over time
        sampling_rate: Sampling rate of the signal in Hz
    
    Returns:
        rr_intervals: Array of RR intervals in milliseconds
    """
    # Find peaks in the heart rate signal
    peaks, _ = signal.find_peaks(heart_rate_signal, distance=int(sampling_rate * 0.5))
    
    if len(peaks) < 2:
        warnings.warn("Not enough peaks detected for HRV analysis")
        return np.array([])
    
    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(peaks) * (1000 / sampling_rate)
    return rr_intervals

def compute_hrv_metrics(rr_intervals):
    """
    Compute HRV metrics from RR intervals.
    
    Args:
        rr_intervals: Array of RR intervals in milliseconds
    
    Returns:
        dict: Dictionary containing HRV metrics
    """
    if len(rr_intervals) < 2:
        return {
            'sdnn': 0,
            'rmssd': 0,
            'valid': False
        }
    
    # SDNN (Standard Deviation of NN intervals)
    sdnn = np.std(rr_intervals)
    
    # RMSSD (Root Mean Square of Successive Differences)
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(rr_diff)))
    
    return {
        'sdnn': round(sdnn, 2),
        'rmssd': round(rmssd, 2),
        'valid': True
    }

def analyze_hrv(heart_rate_signal, sampling_rate):
    """
    Perform complete HRV analysis on the heart rate signal.
    
    Args:
        heart_rate_signal: Array of heart rate values over time
        sampling_rate: Sampling rate of the signal in Hz
    
    Returns:
        dict: Dictionary containing HRV metrics
    """
    rr_intervals = extract_rr_intervals(heart_rate_signal, sampling_rate)
    hrv_metrics = compute_hrv_metrics(rr_intervals)
    return hrv_metrics
