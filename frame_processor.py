import cv2
import numpy as np
from typing import List, Tuple, Generator

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    """Resize frame to target size."""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def process_frames_in_batches(frames: List[np.ndarray], 
                            batch_size: int = 100,
                            target_size: Tuple[int, int] = (320, 240)) -> Generator[np.ndarray, None, None]:
    """
    Process frames in batches to reduce memory usage.
    Returns a generator of processed frame batches.
    """
    current_batch = []
    
    for frame in frames:
        # Resize frame and convert to grayscale to reduce memory
        resized_frame = resize_frame(frame, target_size)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to float32 to reduce memory (vs complex128)
        frame_data = gray_frame.astype(np.float32) / 255.0
        
        current_batch.append(frame_data)
        
        if len(current_batch) == batch_size:
            yield np.array(current_batch)
            current_batch = []
    
    # Yield remaining frames
    if current_batch:
        yield np.array(current_batch)

def compute_batch_fft(batch: np.ndarray, fps: float, freq_min: float, freq_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT for a batch of frames.
    Returns filtered signal and frequencies.
    """
    # Compute FFT
    fft = np.fft.fft(batch.mean(axis=(1, 2)), axis=0)
    frequencies = np.fft.fftfreq(batch.shape[0], d=1.0/fps)
    
    # Apply frequency filter
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    
    filtered_fft = fft.copy()
    filtered_fft[:bound_low] = 0
    filtered_fft[bound_high:-bound_high] = 0
    filtered_fft[-bound_low:] = 0
    
    return filtered_fft, frequencies

def aggregate_batch_results(batch_results: List[Tuple[np.ndarray, np.ndarray]], 
                          fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate results from multiple batches.
    Returns combined filtered signal and frequencies.
    """
    all_ffts = []
    all_freqs = []
    
    for fft, freqs in batch_results:
        all_ffts.extend(fft)
        all_freqs.extend(freqs)
    
    return np.array(all_ffts), np.array(all_freqs)
