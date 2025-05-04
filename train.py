import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from datetime import datetime
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from model import LSTMModel, TimeSeriesDataset
from preprocessing import load_and_preprocess
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Parameters - use simpler model and lower learning rate
csv_path = 'train_FD001_cleaned.csv'
seq_len = 30
n_epochs = 50
batch_size = 32  # Reduced batch size for better stability
learning_rate = 0.0005  # Reduced learning rate for stability
hidden_dim = 64  # Reduced from 128
dropout = 0.1  # Reduced from 0.2

print("Loading and preprocessing data...")
# Load and preprocess data (limit print statements)
X, ids, y = load_and_preprocess(csv_path, seq_len=seq_len)
print(f"Data loaded: {X.shape[0]} sequences, {len(np.unique(ids))} engines")
print(f"Target range: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}")

# Check for NaN in input data
if np.isnan(X).any() or np.isnan(y).any():
    print("Warning: Input data contains NaN values. Fixing...")
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

# Split data into train and validation sets
X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X, y, ids, test_size=0.2, random_state=seed)
print(f"Train set: {X_train.shape[0]} sequences, Val set: {X_val.shape[0]} sequences")

# Set up model, criterion and optimizer
input_dim = X.shape[2]  # Number of features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1, output_dim=1, dropout=dropout).to(device)

# Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

model.apply(init_weights)

class CustomLoss(nn.Module):
    def __init__(self, scale_factor=100.0):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.scale_factor = scale_factor
        
    def forward(self, y_pred, y_true):
        # Scale the predictions to match the target scale
        y_pred_scaled = y_pred * self.scale_factor
        return self.mse_loss(y_pred_scaled, y_true)

# Simple MSE loss with NaN checking and scaling
def safe_mse_loss(y_pred, y_true, scale_factor=100.0):
    # Check for NaN
    if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
        print("Warning: NaN detected in loss computation")
        # Replace NaN with zeros
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
    
    # Scale the predictions to match the target scale
    y_pred_scaled = y_pred * scale_factor
    return nn.MSELoss()(y_pred_scaled, y_true)

criterion = CustomLoss(scale_factor=100.0)  # Use custom loss with scaling
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Create dataset and dataloader with safeguards
def safe_collate(batch):
    try:
        # Standard collate
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        
        # Check for NaN
        if torch.isnan(x).any() or torch.isnan(y).any():
            print("Warning: NaN detected in batch")
            x = torch.nan_to_num(x, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
            
        return x, y
    except Exception as e:
        print(f"Collate error: {e}")
        # Return empty tensors if collation fails
        return torch.tensor([]), torch.tensor([])

train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=safe_collate)

# Track metrics for plotting (minimal)
train_losses = []
val_losses = []

# Training loop with NaN detection and gradient clipping
print("Starting training...")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for X_batch, y_batch in train_loader:
        # Skip empty batches (from safe_collate)
        if len(X_batch) == 0:
            continue
            
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Skip batch if output contains NaN
        if torch.isnan(outputs).any():
            print(f"Warning: NaN in model output at epoch {epoch+1}")
            continue
        
        loss = criterion(outputs, y_batch)
        
        # Skip backprop if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at epoch {epoch+1}, skipping batch")
            continue
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # Check for NaN in gradients
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"Warning: NaN gradient at epoch {epoch+1}, skipping update")
            continue
            
        optimizer.step()
        
        epoch_loss += loss.item() * X_batch.size(0)
        batch_count += 1
    
    # Print epoch statistics (minimal)
    if batch_count > 0:
        epoch_loss = epoch_loss / (batch_count * batch_size)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{n_epochs} - No valid batches")
        continue
    
    # Validation with NaN handling (every 5 epochs to reduce output)
    if (epoch+1) % 5 == 0 or epoch == n_epochs-1:
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            
            try:
                val_preds = model(X_val_tensor).cpu().numpy()
                
                # Check and fix NaN predictions
                if np.isnan(val_preds).any():
                    print("Warning: NaN in validation predictions")
                    val_preds = np.nan_to_num(val_preds, nan=0.0)
                    
                y_val_np = y_val_tensor.cpu().numpy()
                
                # Calculate metrics
                val_mse = mean_squared_error(y_val_np, val_preds)
                val_rmse = math.sqrt(val_mse)
                val_losses.append(val_mse)
                
                print(f"Validation - RMSE: {val_rmse:.4f}, "
                      f"Pred range: {val_preds.min():.1f} to {val_preds.max():.1f}")
                
                # Update learning rate based on validation loss
                scheduler.step(val_mse)
                
            except Exception as e:
                print(f"Validation error: {e}")
                continue

# Final model evaluation
print("\nTraining complete. Evaluating model...")
model.eval()
with torch.no_grad():
    try:
        # Validation predictions
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = model(X_val_tensor).cpu().numpy()
        val_preds = np.nan_to_num(val_preds, nan=0.0)  # Replace any NaNs
        
        # RUL prediction statistics 
        val_rmse = math.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Final validation RMSE: {val_rmse:.4f}")
        print(f"Prediction range: {val_preds.min():.1f} to {val_preds.max():.1f}")
        
        # Save model with timestamp
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"lstm_FD001_{now}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")
        
        # Simple plot of the validation predictions
        plt.figure(figsize=(10, 6))
        plt.plot(y_val[:100], 'bo-', label='True')
        plt.plot(val_preds[:100], 'ro-', label='Predicted')
        plt.title('RUL Predictions vs Actual (First 100 samples)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"val_plot_{now}.png")
        print(f"Validation plot saved as val_plot_{now}.png")
        
    except Exception as e:
        print(f"Evaluation error: {e}")

# Restore warnings
warnings.resetwarnings() 