import numpy as np
from pathlib import Path
import yaml
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from data_compression import load_chunked_dataset

def load_config(config_path='cfg.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calc_partial_correlation(x, y, X_ctrl):
    """
    Calculate partial correlation between x and y, controlling for other variables
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector of shape (n_samples,)
    y : np.ndarray
        Target vector of shape (n_samples,)
    X_ctrl : np.ndarray
        Control matrix of shape (n_samples, n_controls)
        If no control variables, should be None
    
    Returns:
    --------
    float
        Partial correlation coefficient
    """
    if X_ctrl is None or X_ctrl.shape[1] == 0:
        return pearsonr(x, y)[0]
    
    # Ensure X_ctrl is 2D
    if len(X_ctrl.shape) == 1:
        X_ctrl = X_ctrl.reshape(-1, 1)
    
    # Residualize x
    x_resid = x - X_ctrl @ np.linalg.lstsq(X_ctrl, x[:, np.newaxis], rcond=None)[0].flatten()
    
    # Residualize y
    y_resid = y - X_ctrl @ np.linalg.lstsq(X_ctrl, y[:, np.newaxis], rcond=None)[0].flatten()
    
    return pearsonr(x_resid, y_resid)[0]

def calc_objective_score(X, y, selected_indices, n_jobs=1):
    """
    Calculate objective score based on correlation for temporal window selection
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix with all candidate lags
    y : ndarray
        Target variable
    selected_indices : list
        Indices of selected features to evaluate
    n_jobs : int
        Number of CPU cores to use
    
    Returns:
    --------
    float
        Objective score
    """
    if len(selected_indices) == 0:
        return float('-inf')
    
    # Use only selected features
    X_selected = X[:, selected_indices]
    
    # Simple correlation-based score
    try:
        # Calculate direct correlations with target
        correlations = []
        for i in range(X_selected.shape[1]):
            # Check if either array is constant
            x_col = X_selected[:, i]
            if np.std(x_col) < 1e-8 or np.std(y) < 1e-8:
                continue
                
            # Calculate correlation
            corr = np.corrcoef(x_col, y)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        if not correlations:  # If no valid correlations
            return float('-inf')
            
        # For balanced selection, we want to:
        # 1. Reward high overall correlation
        # 2. Encourage diversity in lags (not just picking the highest correlation)
        
        mean_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        
        # Balance between average and maximum correlation
        # Encourages diversity while still valuing strong predictors
        combined_score = 0.7 * mean_corr + 0.3 * max_corr
        
        # Apply diminishing returns rather than direct penalty for more lags
        # This softly encourages parsimony without being too strict
        size_factor = 1.0 / (1.0 + 0.03 * (len(selected_indices) - 1))
        
        return combined_score * size_factor
        
    except Exception as e:
        print(f"Error in score calculation: {str(e)}")
        return float('-inf')

def optimize_temporal_window(data_dir, config_path='cfg.yml', min_lags=3, max_lags=8):
    """
    Optimize temporal window using forward-backward selection independently for each spatial location
    
    Parameters:
    -----------
    data_dir : str or Path
        Path to data directory
    config_path : str
        Path to config file
    min_lags : int
        Minimum number of lags to select per location
    max_lags : int
        Maximum number of lags to select per location
        
    Returns:
    --------
    list
        Common temporal window indices
    float
        Average objective score
    """
    # Load configuration and data
    config = load_config(config_path)
    data_dir = Path(data_dir)
    chunk_dirs = config['data']['chunk_dirs']
    
    print("Loading high-resolution data...")
    high_res_data = load_chunked_dataset(data_dir / chunk_dirs['high_res'])
    
    # Parameters
    n_jobs = min(8, cpu_count() - 2)  # Use 8 cores or max-2 if less available
    candidate_lags = list(range(-24, 0))  # All hourly lags up to 24 hours
    print(f"\nConsidering {len(candidate_lags)} hourly lags: {candidate_lags}")
    
    # Get spatial dimensions
    h, w = high_res_data[0].shape
    print(f"\nProcessing all {h}x{w} spatial locations")
    
    # Dictionary to track selected lags across all locations
    lag_frequencies = {lag: 0 for lag in candidate_lags}
    valid_locations = 0
    all_scores = []
    min_lag = abs(min(candidate_lags))
    
    # Process each spatial location
    for i in tqdm(range(h), desc="Processing rows"):
        for j in range(w):
            # Prepare time series for this location
            X = []
            y = []
            for t in range(min_lag, len(high_res_data)):
                features = [high_res_data[t + lag][i, j] for lag in candidate_lags]
                X.append(features)
                y.append(high_res_data[t][i, j])
            
            X = np.array(X)
            y = np.array(y)
            
            # Check if data is valid for this location (not constant)
            if np.std(y) < 1e-8 or np.all(np.std(X, axis=0) < 1e-8):
                continue
                
            valid_locations += 1
            
            # Calculate individual correlations for each lag
            individual_scores = []
            for lag_idx in range(len(candidate_lags)):
                score = calc_objective_score(X, y, [lag_idx], n_jobs)
                if not np.isinf(score):
                    individual_scores.append((score, lag_idx))
            
            # Sort by correlation score
            individual_scores.sort(reverse=True)
            
            # Start with top correlated lag
            selected = [individual_scores[0][1]] if individual_scores else []
            if not selected:
                continue
                
            # Forward-greedy selection for remaining lags
            candidate_set = list(range(len(candidate_lags)))
            candidate_set.remove(selected[0])
            best_score = calc_objective_score(X, y, selected, n_jobs)
            
            # Forward pass to select additional lags
            while len(selected) < max_lags and candidate_set:
                scores = []
                for idx in candidate_set:
                    curr_selected = selected + [idx]
                    score = calc_objective_score(X, y, curr_selected, n_jobs)
                    if not np.isinf(score):
                        scores.append((score, idx))
                
                if not scores:
                    break
                    
                best_new_score, best_idx = max(scores)
                if best_new_score >= best_score or len(selected) < min_lags:
                    # Keep adding lags until min_lags is reached, even if score doesn't improve
                    best_score = best_new_score
                    selected.append(best_idx)
                    candidate_set.remove(best_idx)
                else:
                    break
            
            # Backward pass only if we have more than the minimum
            if len(selected) > min_lags:
                improved = True
                while improved and len(selected) > min_lags:
                    improved = False
                    scores = []
                    
                    for k in range(len(selected)):
                        curr_selected = selected[:k] + selected[k+1:]
                        score = calc_objective_score(X, y, curr_selected, n_jobs)
                        if not np.isinf(score):
                            scores.append((score, k))
                    
                    if not scores:
                        break
                        
                    best_new_score, worst_idx = max(scores)
                    if best_new_score >= best_score:
                        best_score = best_new_score
                        selected.pop(worst_idx)
                        improved = True
                    else:
                        break
            
            # Update lag frequencies
            for idx in selected:
                lag_frequencies[candidate_lags[idx]] += 1
            
            if not np.isinf(best_score):
                all_scores.append(best_score)
    
    print(f"\nProcessed {valid_locations} valid locations")
    
    if valid_locations == 0:
        print("Error: No valid locations found. Using default 6-hour window.")
        optimized_window = [-6, -5, -4, -3, -2, -1]
        average_score = 0.0
    else:
        # Find common temporal window based on frequency threshold
        threshold = valid_locations * 0.15  # Select lags that appear in at least 15% of locations
        optimized_window = [lag for lag, freq in lag_frequencies.items() if freq >= threshold]
        optimized_window.sort()  # Sort lags in ascending order
        
        # If no lags meet the threshold or too few, use the top N most frequent lags
        if not optimized_window or len(optimized_window) < min_lags:
            print(f"Warning: Not enough lags met the threshold. Using top {min_lags} most frequent lags.")
            top_lags = sorted(lag_frequencies.items(), key=lambda x: x[1], reverse=True)[:min_lags]
            optimized_window = [lag for lag, _ in top_lags]
            optimized_window.sort()
        
        # If too many lags meet the threshold, limit to top N most frequent
        elif len(optimized_window) > max_lags:
            print(f"Warning: Too many lags met the threshold. Limiting to top {max_lags} most frequent lags.")
            top_lags = sorted([(lag, freq) for lag, freq in lag_frequencies.items() if lag in optimized_window], 
                             key=lambda x: x[1], reverse=True)[:max_lags]
            optimized_window = [lag for lag, _ in top_lags]
            optimized_window.sort()
        
        average_score = np.mean(all_scores) if all_scores else 0.0
    
    # Save results
    print("\nSaving results...")
    with open(data_dir / "temporal_window_optimization.txt", "w") as f:
        f.write("Temporal Window Optimization Results\n")
        f.write("==================================\n\n")
        f.write(f"Common temporal window: {optimized_window}\n")
        f.write(f"Average objective score: {average_score:.4f}\n\n")
        f.write(f"Valid locations analyzed: {valid_locations}/{h*w} ({valid_locations/(h*w)*100:.1f}%)\n\n")
        f.write("Lag frequencies:\n")
        for lag, freq in sorted(lag_frequencies.items()):
            if valid_locations > 0:
                f.write(f"Lag {lag}h: {freq}/{valid_locations} locations ({freq/valid_locations*100:.1f}%)\n")
            else:
                f.write(f"Lag {lag}h: {freq}/0 locations (0.0%)\n")
        f.write("\nExplanation:\n")
        f.write("- Used modified forward-backward selection for each spatial location\n")
        f.write(f"- Selected lags that appear in at least 15% of valid locations\n")
        f.write(f"- Enforced minimum of {min_lags} lags and maximum of {max_lags} lags\n")
        f.write("- Used correlation-based objective score with diversity incentives\n")
        f.write("- Analyzed all spatial locations (no sampling)\n")
        f.write("- Considered lags up to 24 hours\n")
    
    # Update config
    config['regression']['temporal_window'] = optimized_window
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    return optimized_window, average_score

if __name__ == "__main__":
    data_dir = "data"  # Update this if needed
    # Use more permissive parameters to ensure we get diverse lag selections
    optimized_window, score = optimize_temporal_window(
        data_dir, 
        min_lags=3,       # Ensure at least 3 lags are selected per location
        max_lags=8        # Limit to 8 lags maximum per location
    )
    print(f"\nOptimization complete!")
    print(f"Common temporal window: {optimized_window}")
    print(f"Mean objective score: {score:.4f}")
