import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Dataset wrapper for time-series data
class TimeSeriesDataset(Dataset):
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step output
        out = self.fc(out)
        return out.squeeze()
def create_rolling_features(df, window_size=5):
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
def walk_forward_validation(model_class, train_data, train_labels, val_data, val_labels, input_dim, **kwargs):
    model = model_class(input_dim=input_dim, **kwargs)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TimeSeriesDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(10):  # You can tune this
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        X_val = torch.tensor(val_data, dtype=torch.float32)
        y_val = torch.tensor(val_labels, dtype=torch.float32)
        predictions = model(X_val).numpy()
        y_val = y_val.numpy()

    rmse = math.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# Generate sequences from df_engineered
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
def predict_lstm(model, X_data):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        predictions = model(X_tensor).numpy()
    return predictions

if __name__ == "__main__":
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
    col_names = index_names + setting_names + sensor_names

    train_1 = pd.read_csv('train_FD001.csv', header=None, names=col_names)
    train_2 = pd.read_csv('train_FD002.csv', header=None, names=col_names)
    train_3 = pd.read_csv('train_FD003.csv', header=None, names=col_names)
    train_4 = pd.read_csv('train_FD004.csv', header=None, names=col_names)

    #RUL of each sensor

    grouped_data_1=train_1.groupby('unit_number')

    grouped_data_2=train_2.groupby('unit_number')

    grouped_data_3=train_3.groupby('unit_number')

    grouped_data_4=train_4.groupby('unit_number')

    time_cycles_average=train_1[index_names].groupby('unit_number').mean()

    max_time_cycles=train_1[index_names].groupby('unit_number').max()['time_cycles']

    merged = train_1.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1)

    df=merged
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[sensor_names + setting_names])
    df_scaled = df.copy()
    df_scaled[sensor_names + setting_names] = scaled_features

    df_engineered, new_features = create_rolling_features(df_scaled)

    # Calculate degradation rates
    for sensor in sensor_names:
        df_engineered[f'{sensor}_degradation_rate'] = df_engineered.groupby('unit_number')[sensor].diff()

    # Time-based features
    df_engineered['time_since_start'] = df_engineered.groupby('unit_number')['time_cycles'].transform('min')
    df_engineered['time_to_failure'] = df_engineered.groupby('unit_number')['time_cycles'].transform('max') - df_engineered['time_cycles']

    # Handle missing values in new features
    df_engineered = df_engineered.fillna(method='ffill').fillna(method='bfill')

    # Features to use
    features_for_model = sensor_names + setting_names + new_features + ['time_since_start', 'time_to_failure']

    # Create LSTM-compatible sequences
    X_seq, y_seq = generate_sequences(df_engineered, features=features_for_model, window_size=30)

    print("Shape of input sequences:", X_seq.shape)
    print("Shape of RUL labels:", y_seq.shape)
    train_X, val_X, train_y, val_y = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    print("Train set shape:", train_X.shape)
    print("Validation set shape:", val_X.shape)
    input_dim = train_X.shape[2]  # number of features per timestep

    results = walk_forward_validation(
        LSTMModel,
        train_X, train_y,
        val_X, val_y,
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=1
    )

    print("Evaluation Metrics:", results)
    lstm_model = LSTMModel(input_dim=train_X.shape[2], hidden_dim=64, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    train_dataset = TimeSeriesDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(10):
        lstm_model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = lstm_model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Predict and plot
    val_preds_lstm = predict_lstm(lstm_model, val_X)
    print(f"LSTM Model RMSE: {math.sqrt(mean_squared_error(val_y, val_preds_lstm))}")
    print(f"LSTM Model MAE: {mean_absolute_error(val_y, val_preds_lstm)}")
    print(f"LSTM Model R2: {r2_score(val_y, val_preds_lstm)}")
