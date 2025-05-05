#!/bin/bash

echo "Preparing environment for AeroHub RUL Predictor..."

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip install -q streamlit torch pandas numpy matplotlib scikit-learn

# Install optional packages for advanced models (if needed)
echo "Installing optional packages for advanced models..."
pip install -q tensorflow lightgbm joblib

# Run the Streamlit app
echo "Starting AeroHub RUL Predictor..."
streamlit run app.py 