# AeroHub RUL Predictor - LSTM Model

This application uses LSTM (Long Short-Term Memory) neural networks to predict the Remaining Useful Life (RUL) of engines based on sensor data.

## Features

- Interactive web interface built with Streamlit
- LSTM-based prediction for time-series sensor data
- Support for multiple input file formats
- Detailed visualization of predictions and results
- Comparative analysis of actual vs. predicted RUL
- Export capabilities for prediction results

## Getting Started

### Prerequisites

- Python 3.7+
- Dependencies listed in `requirement.txt`

### Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirement.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## How to Use

1. Launch the application using the command above
2. Select a trained model using the dropdown in the sidebar
3. Adjust the sequence length parameter if needed (default: 30)
4. Upload your engine sensor data file (CSV or text format)
5. View the analysis results in the different tabs:
   - RUL Distribution: Statistical view of predicted RUL
   - Engine Predictions: Detailed view of individual engine predictions
   - Data Table: Complete data for all predictions with download options

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

## Model Files

The application uses PyTorch-based LSTM models stored as `.pt` files. Two main types of models are included:

- `lstm_FD001_*.pt`: Model trained on the FD001 dataset
- `lstm_all_datasets_*.pt`: Model trained on multiple datasets for better generalization

The application will automatically detect and list all available models.

## Technical Details

### Data Processing

- Features are normalized using StandardScaler
- Sequence-based approach with adjustable window size
- Missing values are handled automatically

### Model Architecture

- LSTM-based neural network with configurable layers
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
2. Model file is properly trained and saved in PyTorch format
3. All required dependencies are installed
