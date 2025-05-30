# Spatio-Temporal Regression by Data Transformation

This project implements a spatio-temporal regression model to predict high-resolution climate data from low-resolution inputs. The core idea is to transform the data using spatio-temporal sliding windows and then apply a regression model.

## Project Structure

```
.
├── cfg.yml
├── data_compression.py
├── data_transform.py
├── idea.txt
├── requirements.txt
├── SpatioTemporalData.py
├── str.py
├── data/
│   ├── bias_pattern.png
│   ├── bias.nc
│   ├── bias.npy
│   ├── high resolution_temporal_avg.png
│   ├── linear_training_history.npy
│   ├── low resolution_temporal_avg.png
│   ├── model_predictions_avg.png
│   ├── train_X.npy
│   ├── high_res_data/
│   ├── low_res_data/
│   ├── test_X/
│   ├── test_y/
│   ├── train_X/
│   └── train_y/
└── __pycache__/
```

## Files

### 1. `cfg.yml`
This YAML file stores all the configuration parameters for the project. It's divided into sections:
- **`data_generation`**: Parameters for generating synthetic spatio-temporal data, including image size, duration (hours), urban heat island effect parameters (falloff rate, temperature ranges), and daily temperature cycle parameters.
- **`data`**: Specifies the directory for storing data and filenames for various datasets (low resolution, high resolution, bias, transformed training/testing sets).
- **`regression`**: Parameters for the spatio-temporal regression model, such as the spatial window size, temporal window (hours to look back), stride for the sliding window, train-test split ratio, choice of regressor (linear, SVM, Random Forest, Neural Network), and loss function.
- **`neural_network`**: Specific parameters if a neural network regressor is chosen, including hidden layer architecture, dropout rate, learning rate, batch size, epochs, and early stopping patience.
- **`visualization`**: Parameters for generating plots, like figure size, colormap, and DPI.

### 2. `requirements.txt`
Lists all the Python dependencies required to run the project. These include:
- `numpy`
- `scikit-learn`
- `scikit-image`
- `matplotlib`
- `torch` (for neural network models)
- `typing-extensions`
- `PyYAML` (for reading the `cfg.yml` file)
- `tqdm` (for progress bars)

You can install these using:
```bash
pip install -r requirements.txt
```

### 3. `SpatioTemporalData.py`
This script is responsible for generating synthetic spatio-temporal data.
- **`SpatioTemporalDataGenerator` class**:
    - Initializes with parameters from `cfg.yml`.
    - `_generate_base_spatial_pattern()`: Creates a complex spatial pattern representing an urban environment with multiple heat centers, sub-centers, cool spots (water bodies, parks), and temperature anomalies. It incorporates fine-scale texture and noise for realism.
    - `_apply_resolution()`: Simulates low-resolution data by applying smoothing and degradation (e.g., rectangular artifacts) to the high-resolution pattern.
    - `_apply_temporal_patterns()`: Modulates the spatial pattern over time using daily and seasonal temperature cycles.
    - `generate()`: Produces either high-resolution or low-resolution spatio-temporal data for a specified duration.
    - `generate_bias()`: Creates a static bias pattern, representing consistent spatial differences (e.g., due to sensor characteristics or fixed geographical features).
    - `sanity_check()`: Visualizes a sample of the generated data for quick verification.
- The `main()` function in this script orchestrates the generation of low-resolution data, high-resolution data, and the bias pattern, saving them into the `data/` directory (chunked for large datasets). It also saves a visualization of the bias pattern.

### 4. `data_compression.py`
This utility script provides functions for handling large datasets by splitting them into smaller, more manageable chunks.
- **`chunk_dataset()`**: Takes a large NumPy array and splits it into multiple smaller `.npy` files (chunks) based on a specified chunk size in megabytes. It saves these chunks into a specified output directory.
- **`load_chunked_dataset()`**: Loads data that has been previously chunked by `chunk_dataset()`. It reads all chunk files from a directory, concatenates them, and returns a single NumPy array.

### 5. `data_transform.py`
This script performs the core data transformation process, preparing the raw spatio-temporal data for the regression model.
- **`load_config()`**: Loads configuration from `cfg.yml`.
- **`pad_images()`**: Adds zero padding to images.
- **`extract_patches()`**: Efficiently extracts patches from an image using `stride_tricks.sliding_window_view`.
- **`transform_data()`**:
    1. Loads the raw low-resolution, high-resolution, and bias datasets (using `load_chunked_dataset` from `data_compression.py`).
    2. Performs a sequential train-test split on the data.
    3. For both training and testing sets:
        - Iterates through time steps (starting from an index that allows for the full temporal window).
        - For each time step `t`:
            - Collects temporal slices from the low-resolution data based on the `temporal_window` specified in `cfg.yml`. These slices are padded.
            - Appends the (padded) static bias image to these temporal slices.
            - Extracts spatial patches from each of these combined temporal slices using the `spatial_window` and `stride` from `cfg.yml`.
            - Concatenates these patches to form a feature vector `X_t`.
            - The target `y_t` is the corresponding flattened high-resolution data at time `t`.
        - The resulting `X` (feature vectors) and `y` (targets) are saved in chunks to the `data/train_X`, `data/train_y`, `data/test_X`, and `data/test_y` directories.
- This script only needs to be run once when new raw data is generated or when transformation parameters in `cfg.yml` are changed.

### 6. `str.py` (Spatio-Temporal Regression)
This is the main script for training the regression model and evaluating its performance.
- **`load_config()`**: Loads configuration from `cfg.yml`.
- **`init_regressor()`**: Initializes the chosen regression model (Linear Regression, SVR, Random Forest, or a PyTorch-based Neural Network) based on `cfg.yml`.
- **`save_training_history()`**: Saves training metrics (e.g., loss per epoch, R² per batch) to a `.npy` file.
- **`train_model()`**:
    - Loads the transformed training data (`X_train`, `y_train`) from the chunked files.
    - If a neural network is chosen:
        - Sets up the model, loss function (MSE), and optimizer (Adam).
        - Trains the model in batches, tracking loss and implementing early stopping.
        - Saves the training history.
    - For other regressors (Linear, SVM, RF):
        - Trains the model in batches (if the dataset is large) to manage memory.
        - Tracks R² and MSE for each batch.
        - Saves the training history.
- **`predict()`**: Makes predictions on the test set using the trained model.
- **`evaluate()`**:
    - Calculates various evaluation metrics: R², MAE, MSE, RMSE, MBE (Mean Bias Error), PSNR (Peak Signal-to-Noise Ratio), and SSIM (Structural Similarity Index Measure).
    - For image-based metrics (PSNR, SSIM), predictions are reshaped back into their original image dimensions.
    - Saves a plot of the temporal average of the model's predictions (`model_predictions_avg.png`).
- The `main()` function:
    1. Loads training and testing data.
    2. Trains the model.
    3. Makes predictions on the test set.
    4. Evaluates the predictions and prints the metrics.

### 7. `idea.txt`
A text file containing the initial conceptual outline and algorithm design for the spatio-temporal regression approach. It describes the data transformation logic, training process, and evaluation metrics.

### 8. `data/` Directory
This directory stores all the data used and generated by the project.
- **`bias.npy`**: The static bias pattern.
- **`bias_pattern.png`**: Visualization of the bias pattern.
- **`high_res_data/`**: Chunked high-resolution synthetic data.
- **`low_res_data/`**: Chunked low-resolution synthetic data.
- **`train_X/`, `train_y/`**: Chunked transformed training features and targets.
- **`test_X/`, `test_y/`**: Chunked transformed testing features and targets.
- **`*_training_history.npy`**: Files storing the metrics recorded during model training (e.g., `linear_training_history.npy`).
- **`*_temporal_avg.png`**: Visualizations of the temporal average of different datasets (e.g., `model_predictions_avg.png`).

## Workflow

-  **Configuration (`cfg.yml`)**: Define all parameters for data generation, transformation, and regression.
-  **Data Generation**: Run `SpatioTemporalData.py` to produce raw data.
-  **Data Transformation**: Run `data_transform.py` to prepare training/testing sets.
-  **Model Training and Evaluation**: Run `str.py` to train the model and evaluate results.

## Quickstart

After cloning the repository and installing dependencies (`pip install -r requirements.txt`), you can execute:
```bash
# 1. Generate synthetic data
python SpatioTemporalData.py

# 2. Transform data for regression
python data_transform.py

# 3. Train and evaluate regressor
python str.py
```

Alternatively, run the orchestrator script:
```bash
./orchestrator.sh
```

Make sure to have Python 3.8+ and required packages installed as listed in `requirements.txt`.

## Core Concept: Spatio-Temporal Data Transformation

The key idea is to create feature vectors for a regression model by combining spatial and temporal information.
For each target high-resolution pixel at a given time `t`:
-   **Spatial Context**: A small window (e.g., 3x3 pixels) is taken from the low-resolution data around the corresponding location.
-   **Temporal Context**: Similar spatial windows are taken from the low-resolution data at several past time steps (e.g., t-1h, t-2h, t-6h, t-12h, t-24h).
-   **Bias Information**: A corresponding spatial window from the static bias image is also included.

These spatial patches from different time steps and the bias are flattened and concatenated to form a single feature vector. The regressor then learns to map these feature vectors to the single high-resolution pixel value. This process is repeated for all pixels and relevant time steps.

## Regressors

The project supports several regression models:
-   **Linear Regression**: A simple and fast baseline.
-   **Support Vector Regressor (SVR)**: A kernel-based method capable of capturing non-linear relationships.
-   **Random Forest Regressor**: An ensemble method that often provides good performance and robustness.
-   **Neural Network**: A multi-layer perceptron (MLP) implemented in PyTorch, allowing for more complex function approximation.

The choice of regressor and its parameters can be configured in `cfg.yml`.

## Evaluation Metrics

The model's performance is assessed using a comprehensive set of metrics:
-   **R² (R-squared)**: Proportion of variance in the dependent variable predictable from the independent variables.
-   **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.
-   **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values.
-   **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same units as the target variable.
-   **MBE (Mean Bias Error)**: Average bias, indicating if the model tends to overpredict or underpredict.
-   **PSNR (Peak Signal-to-Noise Ratio)**: Ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Used for image quality.
-   **SSIM (Structural Similarity Index Measure)**: Measures the similarity between two images, considering luminance, contrast, and structure.

This detailed README should provide a good overview of the project.
