#!/usr/bin/env bash
set -e

# 1. Generate synthetic data
python SpatioTemporalData.py

# 2. Transform data for regression
python data_transform.py

# 3. Train and evaluate regressor
python str.py
