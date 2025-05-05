# AeroHub RUL Predictor

This application uses advanced deep learning models to predict the Remaining Useful Life (RUL) of engines based on sensor data.

## Features

- Interactive web interface built with Streamlit
- Support for multiple model architectures:
  - LSTM (Long Short-Term Memory)
  - BiLSTM (Bidirectional LSTM)
  - CNN-LSTM (Convolutional Neural Network + LSTM)
  - Transformer
  - MultiHead Attention LSTM (requires TensorFlow)
  - CNN + LightGBM (requires TensorFlow and LightGBM)
- Support for multiple input file formats
- Detailed visualization of predictions and results
- Comparative analysis of actual vs. predicted RUL
- Export capabilities for prediction results

## Getting Started

### Prerequisites

- Python 3.7+
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
# On Unix/Linux/Mac
bash run_app.sh

# Or directly with Streamlit
streamlit run app.py
```

## How to Use

1. Launch the application using the command above
2. Select a model type using the dropdown in the sidebar
3. Select a specific model file for the chosen model type
4. Adjust the sequence length parameter if needed (default: 30)
5. Upload your engine sensor data file (CSV or text format)
6. View the analysis results in the different tabs:
   - Data Table: Complete data for all predictions with download options
   - RUL Distribution: Statistical view of predicted RUL
   - Engine Predictions: Detailed view of individual engine predictions

## Input Data Format

The application accepts two types of input files:

### 26-Column Format (without RUL)

```
engine_id, time_cycle, setting_1, setting_2, setting_3, sensor_1, sensor_2, ..., sensor_21
```

### 27-Column Format (with RUL)

```
engine_id, time_cycle, setting_1, setting_2, setting_3, sensor_1, sensor_2, ..., sensor_21, RUL
```

For example:

```
1,1,-0.0007,-0.0004,100.0,518.67,641.82,1589.7,1400.6,14.62,21.61,554.36,2388.06,9046.19,1.3,47.47,521.66,2388.02,8138.62,8.4195,0.03,392,2388,100.0,39.06,23.419
```

## Supported Model Types

### PyTorch Models

These models are always available and require only PyTorch:

- **LSTM**: Traditional LSTM model for sequence processing
- **BiLSTM**: Bidirectional LSTM that processes sequences in both directions
- **CNN-LSTM**: CNN for feature extraction followed by LSTM for sequence processing
- **Transformer**: Transformer-based model with self-attention mechanisms

### TensorFlow Models (optional)

These models require TensorFlow to be installed:

- **MultiHead Attention LSTM**: LSTM with multi-head attention mechanisms

### Hybrid Models (optional)

These models require both TensorFlow and additional libraries:

- **CNN + LightGBM**: CNN for feature extraction with LightGBM for final regression

## Model Files

The application uses various model files:

- PyTorch models: `.pt` extension (LSTM, BiLSTM, CNN-LSTM, Transformer)
- TensorFlow models: `.keras` extension (MultiHead Attention LSTM, CNN)
- LightGBM models: `.pkl` extension

The application will automatically detect and list all available models by type.

## Technical Details

### Data Processing

- Features are normalized using StandardScaler
- Sequence-based approach with adjustable window size
- Missing values are handled automatically

### Model Architectures

- LSTM-based neural networks with configurable layers
- CNN-based feature extraction for certain models
- Transformer and attention-based mechanisms for complex patterns
- Supports different input dimensions through automatic adaptation
- ReLU activation for non-negative RUL prediction

### Prediction Metrics

When the uploaded file contains actual RUL values, the application will calculate and display:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Troubleshooting

If you encounter any issues with file formats or model compatibility, check:

1. File format follows the expected structure (26 or 27 columns)
2. Model files are properly trained and saved in the correct format
3. All required dependencies are installed (especially for TensorFlow and LightGBM models)
4. For hybrid models, ensure all component models are available

## Dependencies

- **Required**: streamlit, torch, pandas, numpy, matplotlib, scikit-learn
- **Optional**: tensorflow, lightgbm, joblib
