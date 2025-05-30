import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import yaml
from data_compression import chunk_dataset

def load_config(config_path='cfg.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SpatioTemporalDataGenerator:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        
        # Load parameters from config
        self.config = config
        gen_cfg = config['data_generation']
        
        # Parameters for the urban heat island effect
        self.center_temp_range = tuple(gen_cfg['center_temp_range'])
        self.outer_temp_range = tuple(gen_cfg['outer_temp_range'])
        self.falloff_rate = gen_cfg['falloff_rate']
        
        # Temporal parameters
        self.daily_peak_hour = gen_cfg['daily_peak_hour']
        self.daily_low_hour = gen_cfg['daily_low_hour']
        self.seasonal_period = 24 * 30 * 12  # Hours in a year
        
    def _generate_base_spatial_pattern(self, h, w, falloff_rate=None):
        """
        Generate base spatial pattern with multiple urban centers and realistic heat distribution
        """
        # Create meshgrid for x and y coordinates
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x, y)
        
        # Use provided falloff rate or default
        rate = falloff_rate if falloff_rate is not None else self.falloff_rate
        
        # Initialize pattern
        spatial_pattern = np.zeros((h, w))
        
        # Define main urban area boundary (smaller than before)
        urban_boundary = np.exp(-1.5 * (xx**2 + yy**2))
        
        # 1. Create multiple urban centers with more distinct profiles
        centers = [
            (0.0, 0.0, 0.9, 4.5),      # Main city center (reduced intensity)
            (-0.3, 0.2, 0.85, 5.5),    # Secondary business district
            (0.25, -0.15, 0.8, 5.0),   # Industrial area
            (0.1, 0.35, 0.75, 4.0),    # Dense residential area
            (-0.2, -0.3, 0.7, 4.5),    # Shopping/entertainment district
            (0.4, 0.1, 0.65, 5.0),     # Added: Tech hub
            (-0.4, -0.1, 0.7, 4.8)     # Added: Commercial center
        ]
        
        for x_pos, y_pos, intensity, sharpness in centers:
            dist = np.sqrt((xx - x_pos)**2 + (yy - y_pos)**2)
            # Create more concentrated heat centers with individual sharpness
            center_pattern = intensity * np.exp(-sharpness * dist)
            spatial_pattern += center_pattern
        
        # 2. Add smaller sub-centers (local hot spots) with more variation
        np.random.seed(42)
        n_subcenters = 25  # Increased number
        
        for _ in range(n_subcenters):
            # Keep centers within urban boundary but allow some to be further out
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0, 0.8)  # Increased radius
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)
            
            intensity = np.random.uniform(0.2, 0.5)  # Reduced intensity range
            size = np.random.uniform(0.03, 0.08)  # Smaller sizes for more distinct spots            
            # Add some asymmetry to each subcenter
            angle = np.random.uniform(0, 2*np.pi)
            stretch = np.random.uniform(1.3, 2.0)
            rotated_x = (xx-x_pos)*np.cos(angle) + (yy-y_pos)*np.sin(angle)
            rotated_y = -(xx-x_pos)*np.sin(angle) + (yy-y_pos)*np.cos(angle)
            dist_stretched = np.sqrt((rotated_x/stretch)**2 + (rotated_y)**2)
            
            subcenter = intensity * np.exp(-dist_stretched/size)
            spatial_pattern += 0.3 * subcenter  # Reduced contribution
        
        # Modify urban boundary to be less constraining
        spatial_pattern *= (0.6 + 0.4 * urban_boundary)  # Changed weights
        
        # 3. Add water bodies and parks (cool spots)
        # Major water body (river-like)
        river_points = [(0.8, -0.4), (0.4, -0.2), (0.0, 0.0), (-0.3, 0.3), (-0.6, 0.5)]
        water_pattern = np.zeros_like(spatial_pattern)
        
        for i in range(len(river_points)-1):
            x1, y1 = river_points[i]
            x2, y2 = river_points[i+1]
            steps = 20
            for t in np.linspace(0, 1, steps):
                x = x1 + t*(x2-x1)
                y = y1 + t*(y2-y1)
                dist = np.sqrt((xx-x)**2 + (yy-y)**2)
                width = np.random.uniform(0.03, 0.06)
                water_pattern += np.exp(-dist/width)
        
        # Add lakes and parks
        n_cool_spots = 20
        for _ in range(n_cool_spots):
            # Distribute throughout the area
            x_pos = np.random.uniform(-0.8, 0.8)
            y_pos = np.random.uniform(-0.8, 0.8)
            
            size = np.random.uniform(0.04, 0.12)
            cooling = np.random.uniform(0.2, 0.4)
            
            dist = np.sqrt((xx - x_pos)**2 + (yy - y_pos)**2)
            cool_spot = np.exp(-dist/size)
            water_pattern += cool_spot
        
        # Apply water cooling effect
        water_pattern = gaussian_filter(water_pattern, sigma=1.0)
        spatial_pattern -= 0.3 * water_pattern
        
        # Add temperature anomalies
        n_hot_anomalies = 30  # Hot spots in cool regions
        n_cool_anomalies = 25  # Cool spots in hot regions
        
        # Function to create anomalies
        def create_anomaly(xx, yy, center_temp, is_hot=True):
            x_pos = np.random.uniform(-0.9, 0.9)
            y_pos = np.random.uniform(-0.9, 0.9)
            
            # Size and intensity vary based on type
            if is_hot:
                size = np.random.uniform(0.02, 0.05)
                intensity = np.random.uniform(0.4, 0.7)
            else:
                size = np.random.uniform(0.03, 0.08)
                intensity = np.random.uniform(0.3, 0.6)
            
            dist = np.sqrt((xx - x_pos)**2 + (yy - y_pos)**2)
            
            # Create sharper anomalies
            anomaly = intensity * np.exp(-dist/size)
            
            # Only apply where appropriate (hot in cool areas, cool in hot areas)
            mask = center_temp < 0.3 if is_hot else center_temp > 0.7
            return anomaly * mask
        
        # Create anomalies after initial pattern but before final normalization
        # Store the pre-normalized pattern
        center_temp = spatial_pattern.copy()
        center_temp = (center_temp - center_temp.min()) / (center_temp.max() - center_temp.min())
        
        # Add hot spots in cool regions
        for _ in range(n_hot_anomalies):
            hot_spot = create_anomaly(xx, yy, center_temp, is_hot=True)
            spatial_pattern += hot_spot
        
        # Add cool spots in hot regions
        for _ in range(n_cool_anomalies):
            cool_spot = create_anomaly(xx, yy, center_temp, is_hot=False)
            spatial_pattern -= cool_spot
        
        # 4. Add fine-scale texture for high-res detail
        texture = gaussian_filter(np.random.normal(0, 1, (h, w)), sigma=1.0)
        spatial_pattern += 0.05 * texture  # Reduced texture contribution
        
        # 5. Apply urban boundary to constrain heat island (less constraining)
        spatial_pattern *= (0.7 + 0.3 * urban_boundary)
        
        # Add some general noise for natural variation
        noise = gaussian_filter(np.random.normal(0, 1, (h, w)), sigma=4.0)
        spatial_pattern += 0.08 * noise  # Reduced noise contribution
        
        # Normalize to [0, 1]
        spatial_pattern = np.clip(spatial_pattern, 0, None)
        spatial_pattern = (spatial_pattern - spatial_pattern.min()) / (spatial_pattern.max() - spatial_pattern.min())
        
        return spatial_pattern
    
    def _apply_resolution(self, data, res):
        """Apply spatial resolution effect with severe rectangular artifacts"""
        if res == 'low':
            rows, cols = data.shape
            
            # First apply strong smoothing
            smoothed = gaussian_filter(data, sigma=5.0)  # Increased initial smoothing
            
            # Create rectangular degradation with more extreme aspect ratio
            vertical_kernel = 16   # Increased from 12
            horizontal_kernel = 8   # Decreased from 6
            
            # Create initial downsampled version
            downsampled_v = np.zeros((rows//vertical_kernel, cols))
            
            # First downsample vertically with more noise
            for i in range(0, rows-vertical_kernel+1, vertical_kernel):
                for j in range(cols):
                    strip = smoothed[i:i+vertical_kernel, j]
                    avg = np.mean(strip) + np.random.normal(0, 0.1)  # Increased noise
                    downsampled_v[i//vertical_kernel, j] = avg
            
            # Double smoothing between vertical and horizontal passes
            upscaled_v = resize(downsampled_v, (rows, cols), order=1, mode='edge')  # Changed to edge mode
            upscaled_v = gaussian_filter(upscaled_v, sigma=2.0)  # Added extra smoothing
            
            # Now downsample horizontally with even smaller kernel
            downsampled_h = np.zeros((rows, cols//horizontal_kernel))
            for i in range(rows):
                for j in range(0, cols-horizontal_kernel+1, horizontal_kernel):
                    strip = upscaled_v[i, j:j+horizontal_kernel]
                    avg = np.mean(strip) + np.random.normal(0, 0.1)  # Increased noise
                    downsampled_h[i, j//horizontal_kernel] = avg
            
            # Upscale horizontally with more aggressive interpolation
            degraded = resize(downsampled_h, (rows, cols), order=1, mode='edge')  # Changed to edge mode
            
            # Add stronger local blending
            degraded = gaussian_filter(degraded, sigma=2.0)  # Increased final smoothing
            
            # Add stronger low frequency variations
            low_freq = gaussian_filter(np.random.normal(0, 1, data.shape), sigma=20.0)  # Increased sigma
            degraded += 0.2 * low_freq  # Increased contribution
            
            # Add some additional structured noise
            structured_noise = gaussian_filter(np.random.normal(0, 1, data.shape), sigma=3.0)
            degraded += 0.1 * structured_noise
            
            # Ensure values stay in [0,1] range
            degraded = np.clip(degraded, 0, 1)
            
            return degraded
        else:
            # For high resolution, apply minimal smoothing but preserve local variations
            return gaussian_filter(data, sigma=0.5)
    
    def _apply_temporal_patterns(self, base_pattern, hour_idx, hr):
        """Apply daily and seasonal temporal patterns"""
        # Daily pattern (24-hour cycle) with sharper peak
        hour_in_day = hour_idx % 24
        daily_factor = np.exp(-0.3 * ((hour_in_day - self.daily_peak_hour) % 24)**2 / 50)
        
        # Seasonal pattern (yearly cycle) with more pronounced seasons
        day_of_year = (hour_idx // 24) % 365
        seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Combine patterns with more weight on daily cycle
        return base_pattern * daily_factor * seasonal_factor
    
    def generate(self, h, w, hr, res, falloff_rate=None):
        """
        Generate spatiotemporal temperature data
        
        Parameters:
        -----------
        h, w : int
            Height and width of the images
        hr : int
            Number of hours to generate
        res : str
            Spatial resolution ('low' or 'high')
        falloff_rate : float, optional
            Rate of exponential decay for urban heat island effect.
            Higher values = faster temperature drop with distance.
        """
        if not isinstance(h, int) or not isinstance(w, int) or not isinstance(hr, int):
            raise ValueError("h, w, and hr must be integers")
        if res not in ['low', 'high']:
            raise ValueError("res must be either 'low' or 'high'")
        
        # Generate base spatial pattern
        base_pattern = self._generate_base_spatial_pattern(h, w, falloff_rate)
        
        # Generate time series
        data = np.zeros((hr, h, w))
        for t in range(hr):
            # Get temporal pattern
            temp_pattern = self._apply_temporal_patterns(base_pattern, t, hr)
            # Apply resolution effects
            data[t] = self._apply_resolution(temp_pattern, res)
        
        return data
    
    def generate_bias(self, h, w):
        """
        Generate a high-resolution bias image
        
        Parameters:
        -----------
        h, w : int
            Height and width of the image
            
        Returns:
        --------
        np.ndarray
            Array of shape (h, w) containing the bias pattern
        """
        # Create random noise
        noise = np.random.normal(0, 1, (h, w))
        # Add some structure using multiple frequency components
        freq_components = []
        for scale in [2, 4, 8, 16]:
            component = gaussian_filter(np.random.normal(0, 1, (h, w)), sigma=scale)
            freq_components.append(component)
        
        # Combine components with different weights
        bias = sum(w * c for w, c in zip([0.4, 0.3, 0.2, 0.1], freq_components))
        # Normalize to [0, 1]
        bias = (bias - bias.min()) / (bias.max() - bias.min())
        
        return bias
    
    def sanity_check(self, data, title=""):
        """Perform sanity checks on the generated data"""
        vis_cfg = self.config['visualization']
        
        # 1. Check value range
        print(f"{title} Value range: [{data.min():.3f}, {data.max():.3f}]")
        
        # 2. Temporal average
        temporal_avg = np.mean(data, axis=0)
        plt.figure(figsize=tuple(vis_cfg['figure_size']))
        plt.imshow(temporal_avg, cmap=vis_cfg['colormap'])
        plt.colorbar(label='Average Temperature')
        plt.title(f"{title} Temporal Average")
        plt.savefig(f"{self.config['data']['directory']}/{title.lower()}_temporal_avg.png",
                   dpi=vis_cfg['dpi'])
        plt.close()
        
        # 3. Check temporal correlations
        if data.shape[0] >= 24:  # If we have at least 24 hours
            noon_temps = data[14::24].mean(axis=(1,2))  # 2 PM temperatures
            midnight_temps = data[2::24].mean(axis=(1,2))  # 2 AM temperatures
            print(f"{title} Average noon temperature: {noon_temps.mean():.3f}")
            print(f"{title} Average midnight temperature: {midnight_temps.mean():.3f}")
        
        # 4. Spatial correlation check
        center_region = data[:, data.shape[1]//3:2*data.shape[1]//3, 
                           data.shape[2]//3:2*data.shape[2]//3]
        outer_region = data[:, :data.shape[1]//4, :data.shape[2]//4]
        print(f"{title} Average center temperature: {center_region.mean():.3f}")
        print(f"{title} Average outer region temperature: {outer_region.mean():.3f}")
        
        # 5. PSNR between consecutive timesteps
        psnr_values = []
        for i in range(data.shape[0]-1):
            psnr_val = psnr(data[i], data[i+1], data_range=1.0)
            psnr_values.append(psnr_val)
        print(f"{title} Average PSNR between consecutive frames: {np.mean(psnr_values):.2f}")

def main():
    # Load configuration
    config = load_config()
    
    # Create data directory if it doesn't exist
    data_dir = Path(config['data']['directory'])
    data_dir.mkdir(exist_ok=True)
    
    # Initialize generator with config
    generator = SpatioTemporalDataGenerator(config)
    
    # Get parameters from config
    h, w = config['data_generation']['image_size']
    hr = config['data_generation']['hours']
    chunk_dirs = config['data']['chunk_dirs']
    
    # Generate and save low resolution data
    print("Generating low resolution data...")
    low_res_data = generator.generate(h, w, hr, 'low')
    low_res_dir = data_dir / chunk_dirs['low_res']
    chunk_dataset(low_res_data, low_res_dir)
    generator.sanity_check(low_res_data, "Low Resolution")
    
    # Generate and save high resolution data
    print("\nGenerating high resolution data...")
    high_res_data = generator.generate(h, w, hr, 'high')
    high_res_dir = data_dir / chunk_dirs['high_res']
    chunk_dataset(high_res_data, high_res_dir)
    generator.sanity_check(high_res_data, "High Resolution")
    
    # Generate and save bias pattern
    print("\nGenerating bias pattern...")
    bias = generator.generate_bias(h, w)
    
    # Visualize and save bias pattern
    plt.figure(figsize=tuple(config['visualization']['figure_size']))
    plt.imshow(bias, cmap=config['visualization']['colormap'])
    plt.colorbar(label='Bias Value')
    plt.title("Bias Pattern")
    plt.savefig(data_dir / config['data']['bias_pattern'], dpi=config['visualization']['dpi'])
    plt.close()
    
    # Save bias in chunks
    bias_data = np.expand_dims(bias, 0)  # Add time dimension to match other data format
    bias_dir = data_dir / "bias_data"
    chunk_dataset(bias_data, bias_dir)

if __name__ == "__main__":
    main()