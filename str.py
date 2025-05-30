import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from pathlib import Path
import numpy.lib.stride_tricks as stride_tricks
from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

def load_config(config_path: str = "cfg.yml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class SpatioTemporalRegressor:
    def __init__(self, config: dict = None):
        """
        Initialize the Spatiotemporal Regressor
        """
        if config is None:
            config = load_config()
        
        self.config = config
        model_config = config['model']
        
        # Spatial and temporal window parameters
        self.wh = model_config['spatial_window_height']
        self.ww = model_config['spatial_window_width']
        self.temporal_window = model_config['temporal_window']
        self.stride = model_config['stride']
        
        # Model parameters
        self.regressor_type = model_config['regressor']
        self.loss_type = model_config['loss']
        self.chunk_size = model_config['chunk_size']
        
        # Device configuration
        if model_config['neural_network']['device'] == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = model_config['neural_network']['device']
        
        self.model = None
        
    def _init_regressor(self, input_dim: int):
        """Initialize the chosen regressor"""
        model_config = self.config['model']
        
        if self.regressor_type == 'linear':
            return LinearRegression()
        elif self.regressor_type == 'svm':
            svm_config = model_config['svm']
            return SVR(
                kernel=svm_config['kernel'],
                C=svm_config['C'],
                gamma=svm_config['gamma']
            )
        elif self.regressor_type == 'rf':
            rf_config = model_config['random_forest']
            return RandomForestRegressor(
                n_estimators=rf_config['n_estimators'],
                max_depth=rf_config['max_depth'],
                min_samples_split=rf_config['min_samples_split'],
                random_state=rf_config['random_state']
            )
        elif self.regressor_type == 'nn':
            # Neural network with config parameters
            nn_config = model_config['neural_network']
            layers = []
            
            # Input layer
            layers.append(nn.Linear(input_dim, nn_config['hidden_layers'][0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(nn_config['dropout_rate']))
            
            # Hidden layers
            for i in range(len(nn_config['hidden_layers']) - 1):
                layers.append(nn.Linear(nn_config['hidden_layers'][i], nn_config['hidden_layers'][i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(nn_config['dropout_rate']))
            
            # Output layer
            layers.append(nn.Linear(nn_config['hidden_layers'][-1], 1))
            
            model = nn.Sequential(*layers).to(self.device)
            return model
        else:
            raise ValueError(f"Unknown regressor type: {self.regressor_type}")
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the specified loss function"""
        if self.loss_type == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif self.loss_type == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.loss_type == 'mbe':
            return np.mean(y_pred - y_true)
        elif self.loss_type == 'psnr':
            return -psnr(y_true, y_pred)  # Negative since we minimize
        elif self.loss_type == 'ssim':
            return -ssim(y_true, y_pred)  # Negative since we minimize
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _pad_images(self, images: np.ndarray) -> np.ndarray:
        """Add zero padding to images"""
        pad_h = self.wh // 2
        pad_w = self.ww // 2
        return np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    def _extract_patches(self, padded_image: np.ndarray) -> np.ndarray:
        """Extract patches using stride tricks for efficiency"""
        windows = stride_tricks.sliding_window_view(
            padded_image, 
            (self.wh, self.ww)
        )[::self.stride, ::self.stride]
        return windows.reshape(-1, self.wh * self.ww)
    
    def transform_data(
        self,
        low_res_data: np.ndarray,
        high_res_data: np.ndarray,
        bias_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data using spatiotemporal sliding windows"""
        train_size = self.config['model']['train_test_split']
        paths_config = self.config['paths']
        
        # Convert to float32 for processing (to avoid precision issues during computation)
        low_res_data = low_res_data.astype(np.float32)
        high_res_data = high_res_data.astype(np.float32)
        bias_image = bias_image.astype(np.float32)
        
        # Sequential train-test split
        split_idx = int(len(low_res_data) * train_size)
        train_low = low_res_data[:split_idx]
        train_high = high_res_data[:split_idx]
        test_low = low_res_data[split_idx:]
        test_high = high_res_data[split_idx:]
        
        def process_dataset(low_res, high_res):
            # Process data in chunks to avoid memory issues
            X_chunks, y_chunks = [], []
            
            for chunk_start in range(24, len(low_res), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(low_res))
                
                # Process current chunk
                X_chunk, y_chunk = [], []
                for t in range(chunk_start, chunk_end):
                    # Get temporal window data
                    temporal_slices = []
                    for offset in self.temporal_window:
                        idx = t + offset
                        if idx >= 0:
                            temporal_slices.append(self._pad_images(low_res[idx:idx+1]))
                    
                    # Add bias image with same padding as temporal slices
                    padded_bias = self._pad_images(bias_image[np.newaxis, :, :])
                    temporal_slices.append(padded_bias)
                    temporal_data = np.concatenate(temporal_slices, axis=0)
                    
                    # Extract patches
                    patches = []
                    for slice_idx in range(len(temporal_data)):
                        patch = self._extract_patches(temporal_data[slice_idx])
                        patches.append(patch)
                    
                    X_t = np.hstack(patches)
                    y_t = high_res[t].flatten()
                    
                    X_chunk.append(X_t)
                    y_chunk.append(y_t)
                
                X_chunks.append(np.vstack(X_chunk))
                y_chunks.append(np.concatenate(y_chunk))
            
            return np.vstack(X_chunks), np.concatenate(y_chunks)
        
        # Transform both train and test sets
        train_X, train_y = process_dataset(train_low, train_high)
        test_X, test_y = process_dataset(test_low, test_high)
        
        # Save transformed data in float16 to save space (if enabled)
        if self.config['experiment']['save_intermediate']:
            data_dir = paths_config['data_dir']
            np.save(f"{data_dir}/{paths_config['train_X_file']}", train_X.astype(np.float16))
            np.save(f"{data_dir}/{paths_config['train_y_file']}", train_y.astype(np.float16))
            np.save(f"{data_dir}/{paths_config['test_X_file']}", test_X.astype(np.float16))
            np.save(f"{data_dir}/{paths_config['test_y_file']}", test_y.astype(np.float16))
        
        return train_X, test_X, train_y, test_y
    
    def train(self, train_X: np.ndarray, train_y: np.ndarray):
        """Train the regressor"""
        input_dim = train_X.shape[1]
        self.model = self._init_regressor(input_dim)
        
        if self.regressor_type == 'nn':
            nn_config = self.config['model']['neural_network']
            
            # Convert to PyTorch tensors
            X = torch.FloatTensor(train_X).to(self.device)
            y = torch.FloatTensor(train_y).to(self.device)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=nn_config['batch_size'], shuffle=True)
            
            # Training loop
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=nn_config['learning_rate'])
            
            self.model.train()
            for epoch in range(nn_config['epochs']):
                total_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0 and self.config['experiment']['verbose']:
                    print(f'Epoch [{epoch+1}/{nn_config["epochs"]}], Loss: {total_loss/len(loader):.4f}')
        else:
            self.model.fit(train_X, train_y)
    
    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Make predictions on test data"""
        if self.regressor_type == 'nn':
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(test_X).to(self.device)
                predictions = self.model(X).cpu().numpy()
            return predictions.squeeze()
        else:
            return self.model.predict(test_X)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> dict:
        """Evaluate model performance using multiple metrics"""
        eval_config = self.config['evaluation']
        paths_config = self.config['paths']
        viz_config = self.config['visualization']
        
        # Reshape predictions and true values into images
        y_true_imgs = y_true.reshape(-1, *original_shape)
        y_pred_imgs = y_pred.reshape(-1, *original_shape)
        
        # Calculate only requested metrics
        metrics = {}
        
        if 'r2' in eval_config['metrics']:
            metrics['r2'] = r2_score(y_true, y_pred)
        if 'mae' in eval_config['metrics']:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        if 'mse' in eval_config['metrics']:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        if 'rmse' in eval_config['metrics']:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        if 'mbe' in eval_config['metrics']:
            metrics['mbe'] = np.mean(y_pred - y_true)
        if 'psnr' in eval_config['metrics']:
            metrics['psnr'] = np.mean([psnr(true, pred, data_range=eval_config['data_range']) 
                               for true, pred in zip(y_true_imgs, y_pred_imgs)])
        if 'ssim' in eval_config['metrics']:
            metrics['ssim'] = np.mean([ssim(true, pred, data_range=eval_config['data_range'])
                               for true, pred in zip(y_true_imgs, y_pred_imgs)])
        
        # Save temporal average plot if enabled
        if eval_config['plot_results']:
            y_pred_avg = np.mean(y_pred_imgs, axis=0)
            plt.figure(figsize=viz_config['figure_size'])
            plt.imshow(y_pred_avg, cmap=viz_config['color_map'])
            plt.colorbar(label='Average Temperature')
            plt.title('Temporal Average of Model Predictions')
            
            plot_path = f"{paths_config['data_dir']}/{paths_config['model_predictions_plot']}"
            plt.savefig(plot_path, dpi=viz_config['dpi'], format=viz_config['save_format'])
            plt.close()
        
        return metrics

def main():
    # Load configuration
    config = load_config()
    paths_config = config['paths']
    
    # Load data from numpy files
    data_dir = paths_config['data_dir']
    low_res_data = np.load(f"{data_dir}/{paths_config['low_res_file']}")
    high_res_data = np.load(f"{data_dir}/{paths_config['high_res_file']}")
    bias = np.load(f"{data_dir}/{paths_config['bias_file']}")
    
    if config['experiment']['verbose']:
        print(f"Data loaded - Low res: {low_res_data.dtype}, High res: {high_res_data.dtype}, Bias: {bias.dtype}")
    
    # Initialize regressor with config
    regressor = SpatioTemporalRegressor(config)
    
    # Transform data
    train_X, test_X, train_y, test_y = regressor.transform_data(
        low_res_data, high_res_data, bias
    )
    
    # Train model
    if config['experiment']['verbose']:
        print(f"\nTraining {config['model']['regressor']} regressor...")
    
    regressor.train(train_X, train_y)
    
    # Make predictions
    if config['experiment']['verbose']:
        print("Making predictions...")
    
    y_pred = regressor.predict(test_X)
    
    # Evaluate
    metrics = regressor.evaluate(
        test_y, y_pred,
        original_shape=(high_res_data.shape[1], high_res_data.shape[2])
    )
    
    # Print metrics
    if config['experiment']['verbose']:
        print(f"\nModel Evaluation Metrics for {config['experiment']['name']}:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()