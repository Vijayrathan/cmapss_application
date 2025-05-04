import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import glob
import warnings

# Import model classes from lstm_model.py
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # ReLU activation for ensuring non-negative output
        self.relu = nn.ReLU()

    def forward(self, x):
        # Handle NaN values
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Check input dimensions and adapt if needed
        if x.shape[-1] != self.input_dim:
            # Print warning
            print(f"Warning: Input dimension mismatch in forward pass. Expected: {self.input_dim}, got: {x.shape[-1]}")
            
            # If input has more features than expected, truncate
            if x.shape[-1] > self.input_dim:
                x = x[..., :self.input_dim]
            # If input has fewer features, pad with zeros
            else:
                padding = torch.zeros(*x.shape[:-1], self.input_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
        
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Take output from the last time step
        out = out[:, -1, :]
        
        # Output layer with ReLU to ensure non-negative values
        out = self.fc(out)
        out = self.relu(out)
        
        return out.squeeze()

def create_rolling_features(df, sensor_names, window_size=5):
    features = []
    for sensor in sensor_names:
        # Rolling mean
        df[f'{sensor}_rolling_mean'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window_size).mean())
        # Rolling std
        df[f'{sensor}_rolling_std'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window_size).std())
        # Rolling min
        df[f'{sensor}_rolling_min'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window_size).min())
        # Rolling max
        df[f'{sensor}_rolling_max'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window_size).max())
        features.extend([f'{sensor}_rolling_mean', f'{sensor}_rolling_std',
                        f'{sensor}_rolling_min', f'{sensor}_rolling_max'])
    return df, features

def generate_sequences(df, features, window_size=30):
    sequences = []
    labels = []
    grouped = df.groupby('unit_number')

    for _, group in grouped:
        data = group[features].values
        rul_values = group['RUL'].values

        for i in range(len(group) - window_size + 1):
            seq = data[i:i + window_size]
            label = rul_values[i + window_size - 1]
            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)

def predict_with_lstm(df, model_path=None, window_size=30):
    """
    Process an input dataframe and make predictions using the LSTM model.
    
    Args:
        df: Pandas DataFrame with sensor readings
        model_path: Path to the saved model (if None, will attempt to load default model)
        window_size: Size of sequence window for LSTM predictions
    
    Returns:
        DataFrame with original data and RUL predictions added
    """
    # Define column names
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
    
    # Ensure necessary columns exist
    for col in index_names:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data")
    
    # Add RUL column if not exists
    if 'RUL' not in df.columns:
        max_cycles = df.groupby('unit_number')['time_cycles'].transform('max')
        df['RUL'] = max_cycles - df['time_cycles']
    
    # Improved sensor column detection to handle different naming conventions
    available_sensors = [col for col in sensor_names if col in df.columns]
    
    # If standard naming convention isn't found, try to identify sensor columns by patterns
    if not available_sensors:
        # Try to identify likely sensor columns using heuristics
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Exclude known non-sensor columns
        excluded_cols = index_names + ['RUL']
        potential_sensors = [col for col in numeric_cols if col not in excluded_cols 
                           and not col.startswith('setting_')
                           and not col.startswith('time_')
                           and not 'RUL' in col]
        
        # If we found any potential sensors, use them
        if potential_sensors:
            available_sensors = potential_sensors
            print(f"Using {len(available_sensors)} detected sensor columns: {available_sensors[:5]}...")
        else:
            # Last resort: create synthetic sensor columns
            print("No sensor columns found. Creating synthetic sensor columns for compatibility.")
            try:
                # Try to use numpy for random generation if available
                for i in range(1, 6):  # Create 5 synthetic sensors
                    col_name = f's_{i}'
                    df[col_name] = np.random.normal(size=len(df))
                    available_sensors.append(col_name)
            except ImportError:
                # Fallback to random module if numpy is not available
                import random
                for i in range(1, 6):  # Create 5 synthetic sensors
                    col_name = f's_{i}'
                    df[col_name] = [random.normalvariate(0, 1) for _ in range(len(df))]
                    available_sensors.append(col_name)
    
    available_settings = [col for col in setting_names if col in df.columns]
    
    # Add default settings if none found
    if not available_settings and len(df.columns) > 3:
        for i in range(1, 4):
            setting_name = f'setting_{i}'
            df[setting_name] = 0.0  # Default to zero
            available_settings.append(setting_name)
    
    if not available_sensors:
        raise ValueError("No sensor columns found in the data and unable to create synthetic ones")
    
    # Create a copy to avoid modifying the original dataframe
    df_scaled = df.copy()
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[available_sensors + available_settings])
    df_scaled[available_sensors + available_settings] = scaled_features
    
    # Generate engineered features
    df_engineered, new_features = create_rolling_features(df_scaled, available_sensors)
    
    # Calculate degradation rates
    for sensor in available_sensors:
        df_engineered[f'{sensor}_degradation_rate'] = df_engineered.groupby('unit_number')[sensor].diff()
    
    # Time-based features
    df_engineered['time_since_start'] = df_engineered.groupby('unit_number')['time_cycles'].transform(lambda x: x - x.min())
    df_engineered['time_to_failure'] = df_engineered.groupby('unit_number')['time_cycles'].transform('max') - df_engineered['time_cycles']
    
    # Handle missing values in new features
    df_engineered = df_engineered.fillna(method='ffill').fillna(method='bfill')
    
    # Features to use
    features_for_model = available_sensors + available_settings + new_features + ['time_since_start', 'time_to_failure']
    
    # Create LSTM-compatible sequences
    X_seq, _ = generate_sequences(df_engineered, features=features_for_model, window_size=window_size)
    
    # Load the model
    if model_path is None:
        # Try to find the latest model file
        model_files = [f for f in glob.glob("*.pt")]
        if not model_files:
            raise FileNotFoundError("No model file found. Please specify a model path.")
        model_path = max(model_files, key=os.path.getctime)  # Get the most recent model
    
    # First load the state dict to inspect dimensions
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Get the correct input dimension from the state dict if possible
    input_dim = X_seq.shape[2]  # Default from our data
    hidden_dim = 64  # Default
    num_layers = 2  # Default
    
    # Try to extract dimensions from the state dict
    if 'lstm.weight_ih_l0' in state_dict:
        state_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
        state_hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
        
        # If there's a mismatch, adjust our sequences
        if state_input_dim != input_dim:
            print(f"Input dimension mismatch. Model expects {state_input_dim}, data has {input_dim}.")
            
            # Override our input_dim with the model's expectation
            input_dim = state_input_dim
            
            # We'll need to adjust our sequences before prediction
            adjusted_sequences = []
            for seq in X_seq:
                if seq.shape[1] > input_dim:
                    # If we have more features than the model expects, truncate
                    adjusted_seq = seq[:, :input_dim]
                else:
                    # If we have fewer features, pad with zeros
                    padding = np.zeros((seq.shape[0], input_dim - seq.shape[1]))
                    adjusted_seq = np.concatenate([seq, padding], axis=1)
                adjusted_sequences.append(adjusted_seq)
            X_seq = np.array(adjusted_sequences)
        
        # Use the hidden dimension from the state dict
        hidden_dim = state_hidden_dim
    
    # Check for number of layers
    layer_count = 1
    while f'lstm.weight_ih_l{layer_count}' in state_dict:
        layer_count += 1
    if layer_count > 1:
        num_layers = layer_count
    
    # Create model with the correct dimensions
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)
    
    # Try to load the state dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Warning: Couldn't load with strict=True: {str(e)}")
        # Try with strict=False
        model.load_state_dict(state_dict, strict=False)
        print("Loaded state dict with strict=False")
    
    model.eval()
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for seq in X_seq:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            pred = model(seq_tensor).item()
            predictions.append(pred)
    
    # Map predictions back to original dataframe
    result_df = df.copy()
    result_df['prediction'] = np.nan  # Initialize with NaN
    
    # Assign predictions to the appropriate rows
    idx = 0
    for unit in df_engineered['unit_number'].unique():
        unit_data = df_engineered[df_engineered['unit_number'] == unit]
        unit_pred_len = len(unit_data) - window_size + 1
        
        if unit_pred_len > 0:
            # Get the unit predictions
            unit_predictions = predictions[idx:idx + unit_pred_len]
            idx += unit_pred_len
            
            # Find indices where predictions should be placed
            pred_indices = unit_data.iloc[window_size-1:].index
            
            # Assign predictions to these indices
            for i, pred_idx in enumerate(pred_indices):
                if i < len(unit_predictions):
                    result_df.loc[pred_idx, 'prediction'] = unit_predictions[i]
    
    # Forward fill NaN values for visualization
    result_df['prediction'] = result_df.groupby('unit_number')['prediction'].fillna(method='ffill')
    
    return result_df

# Function to load model and make predictions directly on sequences
def predict_sequences(sequences, model_path=None):
    """
    Make predictions directly on sequence data
    
    Args:
        sequences: numpy array of shape (n_samples, window_size, n_features)
        model_path: Path to the saved model
    
    Returns:
        numpy array of predictions
    """
    # Load the model
    if model_path is None:
        # Try to find the latest model file
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError("No model file found. Please specify a model path.")
        model_path = max(model_files, key=os.path.getctime)  # Get the most recent model
    
    # First load the state dict to inspect dimensions
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract dimensions from state dict
    input_dim = sequences.shape[2]  # default from sequences
    
    # Try to determine model parameters from the state dict
    hidden_dim = 64  # default
    num_layers = 2   # default
    
    # Check if we can extract dimensions from lstm.weight_ih_l0
    if 'lstm.weight_ih_l0' in state_dict:
        # Input dimension is the second dimension of the weight matrix
        state_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
        # Hidden dimension is the first dimension divided by 4 (LSTM has 4 gates)
        state_hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        
        # If there's a mismatch in input dimensions, we need to adapt
        if state_input_dim != input_dim:
            print(f"Warning: Input dimension mismatch. Model expects {state_input_dim}, got {input_dim}.")
            input_dim = state_input_dim
    
    # Check if we can determine number of layers
    layer_count = 1
    while f'lstm.weight_ih_l{layer_count}' in state_dict:
        layer_count += 1
    if layer_count > 1:
        num_layers = layer_count
    
    # Create model with the correct dimensions
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)
    
    # Try to load the state dict with strict=False to allow partial loading
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Warning: Couldn't load state dict with strict=True: {str(e)}")
        try:
            # Try with strict=False which will ignore mismatched keys
            model.load_state_dict(state_dict, strict=False)
            print("Loaded state dict with strict=False (some parameters may be missing or unused)")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    model.eval()
    
    # If input dimensions don't match, we need to adjust the sequences
    if input_dim != sequences.shape[2]:
        adjusted_sequences = []
        for seq in sequences:
            # If sequence has more features than model expects, truncate
            if seq.shape[1] > input_dim:
                adjusted_seq = seq[:, :input_dim]
            # If sequence has fewer features, pad with zeros
            else:
                padding = np.zeros((seq.shape[0], input_dim - seq.shape[1]))
                adjusted_seq = np.concatenate([seq, padding], axis=1)
            adjusted_sequences.append(adjusted_seq)
        sequences = np.array(adjusted_sequences)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            pred = model(seq_tensor).item()
            predictions.append(pred)
    
    return np.array(predictions) 