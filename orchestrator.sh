#!/usr/bin/env bash
set -e

echo "Spatio-Temporal Regression by Data Transformation"
echo "================================================"

# Function to handle errors
handle_error() {
  echo "Error occurred in step: $1"
  exit 1
}

echo "Step 1: Generating synthetic data..."
python SpatioTemporalData.py || handle_error "Data Generation"
echo "✓ Synthetic data generated successfully"

echo "Step 2: Analyzing temporal correlation structure..."
python analyze_lag_correlations.py || handle_error "Correlation Analysis"
echo "✓ Temporal correlation analysis complete"

echo "Step 3: Optimizing temporal window..."
python temporal_window_optimizer.py || handle_error "Temporal Window Optimization"
echo "✓ Temporal window optimized successfully"

echo "Step 4: Transforming data for regression..."
python data_transform.py || handle_error "Data Transformation"
echo "✓ Data transformation complete"

echo "Step 5: Training and evaluating regressor..."
python str.py || handle_error "Regression Training"
echo "✓ Model training and evaluation complete"

echo ""
echo "All steps completed successfully!"
