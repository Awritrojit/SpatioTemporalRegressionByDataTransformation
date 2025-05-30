import numpy as np
from pathlib import Path
import math
import yaml
import shutil

def load_config(config_path='cfg.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class Quantizer:
    """Handles data quantization for different precisions"""
    def __init__(self, precision: str, dynamic_range: bool = True, min_val: float = 0.0, max_val: float = 1.0):
        self.precision = precision
        self.dynamic_range = dynamic_range
        self.min_val = min_val
        self.max_val = max_val
        
        # Define dtype and scale factors for different precisions
        self.dtype_map = {
            'fp32': np.float32,
            'fp16': np.float16,
            'fp8': np.uint8  # We'll implement custom 8-bit float quantization
        }
        
        if precision not in self.dtype_map:
            raise ValueError(f"Unsupported precision: {precision}")
        
        self.dtype = self.dtype_map[precision]
    
    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Quantize data to specified precision"""
        if self.dynamic_range:
            self.min_val = float(data.min())
            self.max_val = float(data.max())
        
        if self.precision == 'fp8':
            # Custom 8-bit float quantization
            # Scale to [0, 255] range
            scaled = 255 * (data - self.min_val) / (self.max_val - self.min_val)
            return np.clip(scaled, 0, 255).astype(np.uint8)
        else:
            # For fp16 and fp32, just convert dtype
            return data.astype(self.dtype)
    
    def dequantize(self, data: np.ndarray) -> np.ndarray:
        """Dequantize data back to full precision"""
        if self.precision == 'fp8':
            # Reverse 8-bit quantization
            return self.min_val + (data.astype(np.float32) / 255.0) * (self.max_val - self.min_val)
        else:
            # For fp16 and fp32, just convert back to float32
            return data.astype(np.float32)
    
    def get_metadata(self) -> dict:
        """Get metadata needed for dequantization"""
        return {
            'precision': self.precision,
            'min_val': self.min_val,
            'max_val': self.max_val
        }

def chunk_dataset(data: np.ndarray, output_dir: Path, max_chunk_size_mb: int = 50, config_path='cfg.yml') -> None:
    """
    Save large numpy arrays by splitting them into chunks.
    
    Args:
        data: The numpy array to save
        output_dir: Directory to save chunks in
        max_chunk_size_mb: Maximum size of each chunk in MB
    """
    # Load quantization settings from config
    config = load_config(config_path)
    quant_cfg = config['data']['quantization']
    
    # Create quantizer if enabled
    quantizer = None
    if quant_cfg['enabled']:
        quantizer = Quantizer(
            precision=quant_cfg['precision'],
            dynamic_range=quant_cfg['dynamic_range'],
            min_val=quant_cfg['min_value'],
            max_val=quant_cfg['max_value']
        )
    
    # Remove old chunks directory if exists and create a fresh one
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Save quantization metadata if using quantization
    if quantizer is not None:
        metadata = quantizer.get_metadata()
        np.save(output_dir / 'quantization_metadata.npy', metadata)
    
    # Calculate size of each element in bytes
    dtype = quantizer.dtype if quantizer else data.dtype
    element_size = np.zeros(1, dtype=dtype).nbytes
    
    # Calculate elements per chunk
    elements_per_chunk = (max_chunk_size_mb * 1024 * 1024) // element_size
    
    # Calculate total elements and number of chunks needed
    total_elements = data.size
    num_chunks = int(np.ceil(total_elements / elements_per_chunk))
    
    # Calculate how many timesteps each chunk will contain
    elements_per_timestep = np.prod(data.shape[1:]) if len(data.shape) > 1 else 1
    timesteps_per_chunk = elements_per_chunk // elements_per_timestep
    print(f"Splitting dataset of shape {data.shape} into {num_chunks} chunks...")
    # Report chunk size: use 'timesteps' for time-series data, 'elements' for 1D arrays
    if data.ndim > 1:
        print(f"Each chunk will contain approximately {timesteps_per_chunk} timesteps")
    else:
        print(f"Each chunk will contain approximately {timesteps_per_chunk} elements")
    
    # Split and save data
    for i in range(num_chunks):
        start_idx = i * timesteps_per_chunk
        end_idx = min((i + 1) * timesteps_per_chunk, data.shape[0])
        chunk = data[start_idx:end_idx]
        
        # Quantize if enabled
        if quantizer is not None:
            chunk = quantizer.quantize(chunk)
        
        # Save chunk
        chunk_path = output_dir / f"chunk_{i+1}.npy"
        np.save(chunk_path, chunk)
        chunk_size_mb = chunk.nbytes / (1024 * 1024)
        print(f"Saved chunk {i+1}/{num_chunks} to {chunk_path} (size: {chunk_size_mb:.1f}MB)")

def load_chunked_dataset(chunks_dir: Path, config_path='cfg.yml') -> np.ndarray:
    """
    Load a dataset that was saved in chunks.
    
    Args:
        data_dir: Directory containing the chunks
    
    Returns:
        The complete dataset as a numpy array
    """
    # Load quantization metadata if it exists
    metadata_file = chunks_dir / 'quantization_metadata.npy'
    quantizer = None
    
    if metadata_file.exists():
        metadata = np.load(metadata_file, allow_pickle=True).item()
        quantizer = Quantizer(
            precision=metadata['precision'],
            dynamic_range=False,
            min_val=metadata['min_val'],
            max_val=metadata['max_val']
        )
    else:
        print("No quantization metadata found. Loading without quantization.")
    
    chunks_dir = Path(chunks_dir)
    print(f"Loading chunks from {chunks_dir}...")
    
    # Get all chunk files sorted by number
    chunk_files = sorted(
        chunks_dir.glob("chunk_*.npy"),
        key=lambda x: int(x.stem.split("_")[1])
    )
    
    if not chunk_files:
        raise FileNotFoundError(f"No chunks found in {chunks_dir}")
    
    # Load and dequantize chunks
    chunks = []
    for chunk_file in chunk_files:
        print(f"Loading {chunk_file.name}...")
        chunk = np.load(chunk_file)
        if quantizer is not None:
            chunk = quantizer.dequantize(chunk)
        chunks.append(chunk)
    
    # Concatenate all chunks
    data = np.concatenate(chunks, axis=0)
    print(f"Successfully loaded dataset of shape {data.shape}")
    return data