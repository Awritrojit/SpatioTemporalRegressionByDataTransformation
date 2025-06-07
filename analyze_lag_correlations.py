"""
Analyze correlation structure between different lags in the dataset
to understand the temporal patterns in the data
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from data_compression import load_chunked_dataset

def load_config(config_path='cfg.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_lag_correlations(data_dir, config_path='cfg.yml', sample_rate=0.1):
    """
    Analyze correlation structure between different temporal lags
    
    Parameters:
    -----------
    data_dir : str or Path
        Path to data directory
    config_path : str
        Path to config file
    sample_rate : float
        Fraction of spatial locations to sample (0 to 1)
    """
    # Load configuration and data
    config = load_config(config_path)
    data_dir = Path(data_dir)
    chunk_dirs = config['data']['chunk_dirs']
    
    print("Loading high-resolution data...")
    high_res_data = load_chunked_dataset(data_dir / chunk_dirs['high_res'])
    
    # Parameters
    candidate_lags = list(range(-24, 0))  # All hourly lags up to 24 hours
    print(f"\nAnalyzing {len(candidate_lags)} hourly lags: {candidate_lags}")
    
    # Get spatial dimensions
    h, w = high_res_data[0].shape
    print(f"\nSpatial dimensions: {h}x{w}")
    
    # Prepare arrays to store correlation metrics
    mean_correlations = np.zeros(len(candidate_lags))
    correlation_variance = np.zeros(len(candidate_lags))
    autocorrelation = np.zeros(len(high_res_data) - 1)  # For each time lag from 1 to T-1
    sample_count = 0
    min_lag = abs(min(candidate_lags))
    
    # Sample locations and analyze correlations
    for i in tqdm(range(h), desc="Analyzing rows"):
        for j in range(w):
            # Sample locations based on sample_rate
            if np.random.random() > sample_rate:
                continue
            
            # Extract time series for this location
            time_series = np.array([frame[i, j] for frame in high_res_data])
            
            # Skip locations with constant values
            if np.std(time_series) < 1e-8:
                continue
            
            # Calculate autocorrelation for this location
            for lag in range(1, len(time_series)):
                if lag < len(autocorrelation):
                    corr = np.corrcoef(time_series[lag:], time_series[:-lag])[0, 1]
                    if not np.isnan(corr):
                        autocorrelation[lag-1] += corr
            
            # Calculate correlation with each candidate lag
            location_correlations = []
            for lag_idx, lag in enumerate(candidate_lags):
                # Skip if we don't have enough data points
                if abs(lag) >= len(time_series):
                    continue
                
                # Calculate correlation between current time and lagged time
                target = time_series[min_lag:]
                feature = time_series[min_lag + lag:][:len(target)]
                
                if len(target) > 0 and len(feature) > 0:
                    corr = np.corrcoef(target, feature)[0, 1]
                    if not np.isnan(corr):
                        location_correlations.append((lag_idx, abs(corr)))
            
            # Update mean and variance of correlations
            if location_correlations:
                sample_count += 1
                for lag_idx, corr in location_correlations:
                    mean_correlations[lag_idx] += corr
                    correlation_variance[lag_idx] += corr**2
    
    # Compute final statistics
    if sample_count > 0:
        mean_correlations /= sample_count
        correlation_variance = correlation_variance / sample_count - mean_correlations**2
        autocorrelation /= sample_count
    
    # Create plots directory if it doesn't exist
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot mean correlation by lag
    plt.figure(figsize=(12, 6))
    plt.bar(candidate_lags, mean_correlations, yerr=np.sqrt(correlation_variance), capsize=5,
            color='skyblue', label='Mean absolute correlation')
    plt.xlabel('Lag (hours)')
    plt.ylabel('Mean Absolute Correlation')
    plt.title('Average Correlation by Temporal Lag')
    plt.axhline(y=0.1, color='red', linestyle='--', label='Significance threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'lag_correlations.png', dpi=300)
    
    # Plot autocorrelation function
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(autocorrelation) + 1), autocorrelation, marker='o', linestyle='-', markersize=4)
    plt.xlabel('Lag (hours)')
    plt.ylabel('Autocorrelation')
    plt.title('Temporal Autocorrelation Function')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--')
    # Add vertical lines at multiples of 24 to highlight daily cycles
    for i in range(24, len(autocorrelation), 24):
        plt.axvline(x=i, color='green', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(plots_dir / 'autocorrelation.png', dpi=300)
    
    # Save correlation data
    np.savez(data_dir / 'correlation_analysis.npz',
             lags=candidate_lags,
             mean_correlations=mean_correlations,
             correlation_variance=correlation_variance,
             autocorrelation=autocorrelation)
    
    print("\nCorrelation Analysis Results:")
    print("-----------------------------")
    print("Top 5 most correlated lags:")
    top_lags = sorted(zip(candidate_lags, mean_correlations), key=lambda x: x[1], reverse=True)[:5]
    for lag, corr in top_lags:
        print(f"Lag {lag}h: {corr:.4f}")
    
    print("\nAnalysis complete! Plots saved to the 'plots' directory.")
    return mean_correlations, autocorrelation

if __name__ == "__main__":
    data_dir = "data"
    mean_correlations, autocorrelation = analyze_lag_correlations(data_dir, sample_rate=1.0)  # Use all locations
