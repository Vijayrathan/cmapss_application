import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1, dropout=0.1, scale_factor=1.0):
        super(LSTMModel, self).__init__()
        
        # Simplified architecture for stability
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Simple fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Output scaling factor
        self.scale_factor = scale_factor
        self.input_dim = input_dim
        
    def forward(self, x):
        # Handle NaN input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # LSTM layer
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take only the last time step output
        
        # Output layer with ReLU to ensure non-negative RUL
        out = self.fc(out)
        out = self.relu(out)
        
        # Apply scaling factor if needed
        if self.scale_factor != 1.0:
            out = out * self.scale_factor
            
        return out.squeeze()

def load_model(path, input_dim, device='cpu'):
    # Load the model state dict first to check dimensions
    try:
        state_dict = torch.load(path, map_location=device)
        
        # Determine the input dimension of the saved model
        saved_input_dim = None
        if 'lstm.weight_ih_l0' in state_dict:
            saved_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
        
        # Check if dimensions match
        if saved_input_dim is not None and saved_input_dim != input_dim:
            warnings.warn(f"Input dimension mismatch: saved model has {saved_input_dim} features, "
                          f"but input data has {input_dim} features. Creating a new model with the saved dimensions.")
            input_dim = saved_input_dim
            
        # Create a model with the correct input dimension
        model = LSTMModel(
            input_dim=input_dim, 
            hidden_dim=64, 
            num_layers=1, 
            output_dim=1,
            dropout=0.1,
            scale_factor=100.0  # Scale factor for RUL predictions
        )
        
        # Load the state dict
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")
