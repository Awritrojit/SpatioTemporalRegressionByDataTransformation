import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
from data_compression import load_chunked_dataset

def load_config(config_path='cfg.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_regressor(regressor_type: str, input_dim: int, config: dict):
    """Initialize the chosen regressor"""
    if regressor_type == 'linear':
        return LinearRegression()
    elif regressor_type == 'svm':
        return SVR(kernel='rbf')
    elif regressor_type == 'rf':
        return RandomForestRegressor(n_estimators=100)
    elif regressor_type == 'nn':
        nn_cfg = config['neural_network']
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in nn_cfg['hidden_layers']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(nn_cfg['dropout_rate'])
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        model = nn.Sequential(*layers)
        
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")

def save_training_history(history: dict, model_type: str, config: dict):
    """Save training metrics history to file"""
    data_dir = Path(config['data']['directory'])
    history_file = data_dir / f"{model_type}_training_history.npy"
    np.save(history_file, history)
    print(f"\nTraining history saved to {history_file}")

def train_model(X_train: np.ndarray, y_train: np.ndarray, config: dict):
    """Train the regressor"""
    reg_cfg = config['regression']
    regressor_type = reg_cfg['regressor_type']
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = init_regressor(regressor_type, input_dim, config)
    
    if regressor_type == 'nn':
        nn_cfg = config['neural_network']
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=nn_cfg['learning_rate'])
        
        # Convert to tensors
        X = torch.FloatTensor(X_train)
        y = torch.FloatTensor(y_train)
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=nn_cfg['batch_size'], shuffle=True)
        
        # Initialize history tracking
        history = {
            'epoch': [],
            'loss': [],
            'val_loss': []  # if we implement validation later
        }
        
        # Training loop
        best_loss = float('inf')
        epoch = 0
        
        while True:
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Ask user if they want to continue training
            user_input = input("Continue to iterate? [y/n]: ")
            if user_input.lower() != 'y':
                break
                
            epoch += 1
            
        return model, history

    else:

        print(f"\nTraining {regressor_type} model...")
        batch_size = reg_cfg['batch_size']  # Process 1M samples at a time
        n_batches = int(np.ceil(len(X_train) / batch_size))
        
        # Initialize history tracking
        history = {
            'batch': [],
            'r2': [],
            'mse': []
        }
        
        # Initialize running averages for metrics
        running_r2 = 0
        running_mse = 0
        alpha = 0.1  # Smoothing factor
        
        # Train on sequential batches
        pbar = tqdm(range(n_batches), desc='Training on batches')
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            
            # Get current batch
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Train on current batch
            if i == 0:
                model.fit(X_batch, y_batch)
            else:
                if hasattr(model, 'warm_start'):
                    model.set_params(warm_start=True)
                    model.fit(X_batch, y_batch)
                else:
                    model.fit(X_batch, y_batch)
            
            # Make predictions and compute metrics
            y_pred = model.predict(X_batch)
            r2 = r2_score(y_batch, y_pred)
            mse = mean_squared_error(y_batch, y_pred)
            
            # Update running averages
            if i == 0:
                running_r2 = r2
                running_mse = mse
            else:
                running_r2 = (1 - alpha) * running_r2 + alpha * r2
                running_mse = (1 - alpha) * running_mse + alpha * mse
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'R²': f'{running_r2:.4f}',
                'MSE': f'{running_mse:.6f}'
            })
            
            # Record history
            history['batch'].append(i + 1)
            history['r2'].append(r2)
            history['mse'].append(mse)
        
        # Print final metrics
        print('\nTraining complete!')
        print(f'Average R² Score: {np.mean(history["r2"]):.4f}')
        print(f'Average MSE: {np.mean(history["mse"]):.4f}')
        
        # Save training history
        save_training_history(history, regressor_type, config)
    
    return model

def predict(model, X_test: np.ndarray) -> np.ndarray:
    """Make predictions on test data"""
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X_test)
            if torch.cuda.is_available():
                X = X.cuda()
            predictions = model(X).cpu().numpy()
        return predictions.squeeze()
    else:
        return model.predict(X_test)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, image_size: Tuple[int, int], config: dict) -> dict:
    """Evaluate model predictions and generate visualizations"""
    pixels_per_image = image_size[0] * image_size[1]
    # Ensure we only use complete images (trim incomplete chunks)
    n_complete_samples = min(len(y_true) // pixels_per_image, len(y_pred) // pixels_per_image)
    
    # Trim arrays to have only complete images
    y_true = y_true[:n_complete_samples * pixels_per_image]
    y_pred = y_pred[:n_complete_samples * pixels_per_image]
    
    # Reshape predictions and ground truth back to images
    y_pred_imgs = y_pred.reshape(n_complete_samples, image_size[0], image_size[1])
    y_true_imgs = y_true.reshape(n_complete_samples, image_size[0], image_size[1])
    
    # Calculate temporal averages
    y_pred_avg = np.mean(y_pred_imgs, axis=0)
    y_true_avg = np.mean(y_true_imgs, axis=0)
    
    # Save visualizations
    data_dir = Path(config['data']['directory'])
    viz_cfg = config['visualization']
    
    # Plot predicted and true averages side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(viz_cfg['figure_size']))
    
    im1 = ax1.imshow(y_pred_avg, cmap=viz_cfg['colormap'])
    plt.colorbar(im1, ax=ax1, label='Predicted')
    ax1.set_title("Model Predictions (Temporal Average)")
    
    im2 = ax2.imshow(y_true_avg, cmap=viz_cfg['colormap'])
    plt.colorbar(im2, ax=ax2, label='Ground Truth')
    ax2.set_title("Ground Truth (Temporal Average)")
    
    plt.tight_layout()
    plt.savefig(data_dir / "model_predictions_avg.png", dpi=viz_cfg['dpi'])
    plt.close()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'r2': r2,
        'temporal_avg_pred': y_pred_avg,
        'temporal_avg_true': y_true_avg
    }

def main():
    # Load configuration
    config = load_config()
    data_dir = Path(config['data']['directory'])
    
    # Load complete datasets with debug prints
    print("\nLoading training data...")
    X_train = load_chunked_dataset(data_dir / "train_X")
    print(f"X_train loaded with shape: {X_train.shape}, any NaN: {np.isnan(X_train).any()}")
    
    y_train = load_chunked_dataset(data_dir / "train_y")
    print(f"y_train loaded with shape: {y_train.shape}, any NaN: {np.isnan(y_train).any()}")
    
    print("\nLoading test data...")
    X_test = load_chunked_dataset(data_dir / "test_X")
    print(f"X_test loaded with shape: {X_test.shape}, any NaN: {np.isnan(X_test).any()}")
    
    y_test = load_chunked_dataset(data_dir / "test_y")
    print(f"y_test loaded with shape: {y_test.shape}, any NaN: {np.isnan(y_test).any()}")
    
    print("\nChecking array contents:")
    print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data is empty!")
    
    # Check if shapes match
    if X_train.shape[0] != y_train.shape[0]:
        min_len = min(X_train.shape[0], y_train.shape[0])
        print(f"Warning: trimming X_train and y_train to min length {min_len}")
        X_train = X_train[:min_len]
        y_train = y_train[:min_len]
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train, config)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = predict(model, X_test)
    
    # Get original image shape from config
    image_size = config['data_generation']['image_size']
    
    # Evaluate
    metrics = evaluate(y_test, y_pred, image_size, config)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue  # Skip array metrics (temporal averages)
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()