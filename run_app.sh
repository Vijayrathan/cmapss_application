#!/bin/bash

# Print header
echo "==============================================="
echo "  AeroHub RUL Predictor - Setup and Launch"
echo "==============================================="

# Check Python installation
echo "Checking Python installation..."
python --version

# Check if NumPy is already at the correct version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [[ $? -ne 0 ]]; then
    echo "NumPy not installed. Installing required version..."
    pip install "numpy<2.0.0"
elif [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "NumPy version $NUMPY_VERSION detected. Downgrading to compatible version..."
    pip install "numpy<2.0.0" --force-reinstall
else
    echo "NumPy version $NUMPY_VERSION is compatible."
fi

# Install other dependencies if needed
echo "Checking other dependencies..."
pip install -r requirement.txt

# Launch the application
echo "Launching AeroHub RUL Predictor..."
streamlit run app.py

echo "Application closed." 