import numpy as np
import numpy.lib.stride_tricks as stride_tricks
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import yaml
from data_compression import chunk_dataset, load_chunked_dataset

def load_config(config_path='cfg.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def pad_images(images: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Add zero padding to images"""
    if len(images.shape) == 2:  # Single image
        return np.pad(images, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:  # Batch of images
        return np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

def extract_patches(padded_image: np.ndarray, window_shape: Tuple[int, int], stride: int) -> np.ndarray:
    """Extract patches using stride tricks for efficiency"""
    if len(padded_image.shape) == 3:
        # For 3D array (H, W, T), extract spatial patches while preserving temporal dimension
        H, W, T = padded_image.shape
        h_end = H - window_shape[0] + 1
        w_end = W - window_shape[1] + 1
        windows = stride_tricks.sliding_window_view(
            padded_image[:h_end, :w_end], 
            (*window_shape, T)
        )[::stride, ::stride, 0]  # Take first (and only) temporal slice
        return windows.reshape(-1, window_shape[0] * window_shape[1] * T)
    else:
        # For 2D array
        H, W = padded_image.shape
        h_end = H - window_shape[0] + 1
        w_end = W - window_shape[1] + 1
        windows = stride_tricks.sliding_window_view(
            padded_image[:h_end, :w_end],
            window_shape
        )[::stride, ::stride]
        return windows.reshape(-1, window_shape[0] * window_shape[1])

def transform_data(config_path='cfg.yml') -> None:
    """
    Transform the data using spatiotemporal sliding windows and save in chunks.
    Only needs to be run once when new high/low res data is available.
    """
    # Load configuration
    config = load_config(config_path)
    reg_cfg = config['regression']
    data_cfg = config['data']
    data_dir = Path(data_cfg['directory'])
    chunk_dirs = data_cfg['chunk_dirs']
    
    # Load raw data from chunks
    print("Loading raw datasets...")
    low_res_data = load_chunked_dataset(data_dir / chunk_dirs['low_res'])
    high_res_data = load_chunked_dataset(data_dir / chunk_dirs['high_res'])
    bias_data = load_chunked_dataset(data_dir / "bias_data")
    bias = bias_data[0]  # Remove time dimension
    
    # Get parameters
    spatial_window = tuple(reg_cfg['spatial_window'])
    temporal_window = reg_cfg['temporal_window']
    stride = reg_cfg['stride']
    train_size = reg_cfg['train_size']
    # Calculate padding
    pad_h = spatial_window[0] // 2
    pad_w = spatial_window[1] // 2
    
    # Split data sequentially
    split_idx = int(len(low_res_data) * train_size)
    low_res_train = low_res_data[:split_idx]
    low_res_test = low_res_data[split_idx:]
    high_res_train = high_res_data[:split_idx]
    high_res_test = high_res_data[split_idx:]
    
    # Process training data
    print("\nProcessing training data...")
    
    # Initialize lists to store features and targets
    X_train = []
    y_train = []
    
    # Calculate the minimum time offset for the temporal window
    min_time_idx = abs(min(temporal_window))

    # Extract patches for all valid time steps
    for t in tqdm(range(min_time_idx, len(low_res_train))):
        # Prepare stacked temporal volume with padding
        volume = [pad_images(low_res_train[t+offset], pad_h, pad_w) for offset in temporal_window]
        volume.append(pad_images(bias, pad_h, pad_w))
        stacked = np.stack(volume, axis=-1)  # shape (H+2*pad_h, W+2*pad_w, T)
        H_pad, W_pad, T = stacked.shape
        
        # Loop over spatial positions to extract patches
        for i in range(pad_h, H_pad - pad_h):
            for j in range(pad_w, W_pad - pad_w):
                patch = stacked[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1, :]
                X_train.append(patch.flatten())
                y_train.append(high_res_train[t][i-pad_h, j-pad_w])
    
    # Convert to arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\nSaving training data...")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    chunk_dataset(X_train, data_dir / chunk_dirs['train_x'])
    chunk_dataset(y_train, data_dir / chunk_dirs['train_y'])
    
    # Process test data
    print("\nProcessing test data...")
    
    # Initialize lists to store features and targets
    X_test = []
    y_test = []
    
    # Extract patches for all valid time steps in test set
    for t in tqdm(range(min_time_idx, len(low_res_test))):
        volume = [pad_images(low_res_test[t+offset], pad_h, pad_w) for offset in temporal_window]
        volume.append(pad_images(bias, pad_h, pad_w))
        stacked = np.stack(volume, axis=-1)
        H_pad, W_pad, T = stacked.shape
        for i in range(pad_h, H_pad - pad_h):
            for j in range(pad_w, W_pad - pad_w):
                patch = stacked[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1, :]
                X_test.append(patch.flatten())
                y_test.append(high_res_test[t][i-pad_h, j-pad_w])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nSaving test data...")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    chunk_dataset(X_test, data_dir / chunk_dirs['test_x'])
    chunk_dataset(y_test, data_dir / chunk_dirs['test_y'])

if __name__ == "__main__":
    transform_data()