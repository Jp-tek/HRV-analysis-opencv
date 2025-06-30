from scipy import signal
import numpy as np
from hrv_analysis import analyze_hrv


def find_heart_rate(filtered_signal, frequencies, freq_min, freq_max):
    """
    Calculate heart rate and HRV metrics from the filtered signal.
    Uses memory-efficient processing by working with the already filtered signal.
    """
    # Convert to magnitude spectrum
    magnitude_spectrum = np.abs(filtered_signal)
    
    # Find peaks in the frequency range of interest
    freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    valid_magnitudes = magnitude_spectrum[freq_mask]
    valid_frequencies = frequencies[freq_mask]
    
    if len(valid_magnitudes) == 0:
        return {
            'heart_rate': 0,
            'hrv_metrics': {'sdnn': 0, 'rmssd': 0, 'valid': False}
        }
    
    # Find the dominant frequency (heart rate)
    max_magnitude_idx = np.argmax(valid_magnitudes)
    heart_rate_freq = valid_frequencies[max_magnitude_idx]
    heart_rate = heart_rate_freq * 60
    
    # Calculate HRV metrics using the filtered signal
    sampling_rate = 1.0 / (frequencies[1] - frequencies[0])  # Hz
    hrv_metrics = analyze_hrv(magnitude_spectrum, sampling_rate)
    
    return {
        'heart_rate': round(heart_rate, 1),
        'hrv_metrics': hrv_metrics
    }
