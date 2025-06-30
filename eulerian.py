import numpy as np
from frame_processor import process_frames_in_batches, compute_batch_fft, aggregate_batch_results


# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(video_frames, freq_min, freq_max, fps):
    """
    Apply temporal bandpass filter using FFT with memory-efficient batch processing.
    
    Args:
        video_frames: List of video frames
        freq_min: Minimum frequency to keep
        freq_max: Maximum frequency to keep
        fps: Frames per second
    
    Returns:
        Tuple containing:
        - filtered_signal: The processed signal after FFT filtering
        - frequencies: The frequency array
    """
    batch_results = []
    
    # Process frames in batches
    for frame_batch in process_frames_in_batches(video_frames, batch_size=100):
        # Compute FFT for current batch
        fft, freqs = compute_batch_fft(frame_batch, fps, freq_min, freq_max)
        batch_results.append((fft, freqs))
    
    # Combine results from all batches
    filtered_signal, frequencies = aggregate_batch_results(batch_results, fps)
    
    return filtered_signal, frequencies