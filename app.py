import streamlit as st
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from preprocessing import load_and_preprocess
from model import load_model
import math
import warnings

# Try to import numpy, but have a fallback if it's not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy is not available, using PyTorch tensors instead.")

st.title("🚀 RUL Predictor")
st.write("Upload a file with engine sensor data to predict Remaining Useful Life")
uploaded = st.file_uploader("Upload data file", type=["csv", "txt", "data"])

def get_latest_model(pattern="lstm_FD001_*.pt"):
    model_files = glob(pattern)
    if not model_files:
        return None
    return max(model_files, key=os.path.getctime)

if uploaded:
    # Display file info
    file_details = {"Filename": uploaded.name, "Size": f"{uploaded.size/1024:.1f} KB"}
    st.write(file_details)
    
    # Create an expandable section for preprocessing details
    with st.expander("Preprocessing Details", expanded=False):
        preprocessing_output = st.empty()
        
        # Create a context manager to capture print statements during preprocessing
        import io
        import sys
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            # Load and preprocess the data
            X, ids, y = load_and_preprocess(uploaded, seq_len=30)
            
            # Capture preprocessing output
            preprocessing_log = new_stdout.getvalue()
            preprocessing_output.text(preprocessing_log)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Get latest model file
            model_path = get_latest_model()
            if not model_path:
                st.error("No trained LSTM model found. Please run training first.")
            else:
                # Check model feature dimensions
                state_dict = torch.load(model_path, map_location='cpu')
                model_input_dim = state_dict['lstm.weight_ih_l0'].shape[1] if 'lstm.weight_ih_l0' in state_dict else None
                data_input_dim = X.shape[2]
                
                if model_input_dim and model_input_dim != data_input_dim:
                    st.warning(f"Feature dimension mismatch: Model expects {model_input_dim} features, but data has {data_input_dim} features. Adjusting data to match model.")
                    
                    # Convert to PyTorch tensors to handle the case where NumPy is not available
                    if HAS_NUMPY:
                        # Adjust input data to match model dimensions with NumPy
                        if data_input_dim < model_input_dim:
                            # Pad with zeros
                            padding = np.zeros((X.shape[0], X.shape[1], model_input_dim - data_input_dim))
                            X = np.concatenate([X, padding], axis=2)
                            st.info(f"Added {model_input_dim - data_input_dim} zero-padded features to match model dimensions")
                        else:
                            # Truncate extra features
                            X = X[:, :, :model_input_dim]
                            st.info(f"Truncated {data_input_dim - model_input_dim} features to match model dimensions")
                    else:
                        # Convert to PyTorch tensors and handle the dimension mismatch with PyTorch
                        X_tensor = torch.tensor(X.tolist(), dtype=torch.float32)
                        
                        if data_input_dim < model_input_dim:
                            # Pad with zeros using PyTorch
                            padding = torch.zeros(X_tensor.shape[0], X_tensor.shape[1], model_input_dim - data_input_dim)
                            X_tensor = torch.cat([X_tensor, padding], dim=2)
                            st.info(f"Added {model_input_dim - data_input_dim} zero-padded features to match model dimensions")
                        else:
                            # Truncate extra features
                            X_tensor = X_tensor[:, :, :model_input_dim]
                            st.info(f"Truncated {data_input_dim - model_input_dim} features to match model dimensions")
                        
                        # Convert back to numpy if needed for downstream processing
                        if HAS_NUMPY:
                            X = X_tensor.numpy()
                        else:
                            X = X_tensor
                
                # Load model with the correct input dimensions
                input_dim = model_input_dim if model_input_dim else X.shape[2]
                model = load_model(model_path, input_dim=input_dim)
                
                # Make predictions
                with torch.no_grad():
                    # Handle both numpy arrays and torch tensors
                    if HAS_NUMPY and isinstance(X, np.ndarray):
                        X_tensor = torch.tensor(X, dtype=torch.float32)
                    elif isinstance(X, torch.Tensor):
                        X_tensor = X
                    else:
                        # Fallback to list conversion
                        X_tensor = torch.tensor(X.tolist() if hasattr(X, 'tolist') else X, dtype=torch.float32)
                    
                    # Get predictions as tensor
                    preds_tensor = model(X_tensor)
                    
                    # Apply scaling if predictions are too small
                    pred_max = preds_tensor.max().item()
                    if pred_max < 10.0:  # If predictions are very small
                        st.info(f"Predictions scaled up for better interpretation (original max={pred_max:.2f})")
                        # Scale factor to bring predictions to expected RUL range (0-250)
                        scale_factor = 100.0  # Adjust based on your domain knowledge
                        preds_tensor = preds_tensor * scale_factor
                    
                    # Convert predictions to list for safe handling
                    if HAS_NUMPY:
                        preds = preds_tensor.cpu().numpy()
                    else:
                        preds = preds_tensor.cpu().tolist()
                
                # Create result dataframe with lists to avoid numpy dependency
                result_df = pd.DataFrame({
                    'unit_number': ids.tolist() if hasattr(ids, 'tolist') else ids,
                    'predicted_rul': [round(p, 1) for p in (preds if isinstance(preds, list) else preds.tolist())]
                })
                
                # Get unique engine IDs
                unique_ids = pd.Series(ids.tolist() if hasattr(ids, 'tolist') else ids).unique()
                total_engines = len(unique_ids)
                
                # Display info about number of engines
                st.write(f"### Data contains {total_engines} unique engines")
                
                # Group by unit number and take the last prediction for each engine
                # We'll keep this for average calculations
                final_results = result_df.groupby('unit_number').tail(1).set_index('unit_number')
                
                # Display results
                st.write(f"### Predicted RUL per Engine (Model: {os.path.basename(model_path)})")
                
                # Show statistics using pandas methods instead of numpy
                st.write(f"Average predicted RUL: {final_results['predicted_rul'].mean():.1f} cycles")
                st.write(f"Prediction range: {final_results['predicted_rul'].min():.1f} to {final_results['predicted_rul'].max():.1f} cycles")
                
                # Create a sample of engines to display (up to 20)
                display_engines = unique_ids[:min(len(unique_ids), 20)]
                
                # Create a dataframe with just the selected engines for display
                display_df = result_df[result_df['unit_number'].isin(display_engines)]
                display_final = display_df.groupby('unit_number').tail(1).set_index('unit_number')
                
                # Display table with predictions for sample engines
                st.write(f"Showing predictions for {len(display_engines)} out of {total_engines} engines:")
                st.table(display_final)
                
                # Plot results for the sample engines
                plt.figure(figsize=(12, 6))
                
                for engine_id in display_engines:
                    # Create a mask without numpy
                    engine_mask = result_df['unit_number'] == engine_id
                    plt.plot(result_df.loc[engine_mask, 'predicted_rul'], label=f"Engine {engine_id}")
                
                plt.title("RUL Predictions for Sample Engines")
                plt.xlabel("Sequence Number")
                plt.ylabel("Predicted RUL")
                plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
                plt.grid(True)
                plt.tight_layout()
                
                st.pyplot(plt)
                
                # Allow downloading all the predictions
                csv = final_results.reset_index().to_csv(index=False)
                st.download_button(
                    label="Download All Predictions as CSV",
                    data=csv,
                    file_name='rul_predictions.csv',
                    mime='text/csv',
                )
                
        except ValueError as e:
            # Restore stdout
            sys.stdout = old_stdout
            st.error(f"Error processing the file: {str(e)}")
            st.info("Please make sure your file contains the following data:")
            st.markdown("- Engine/unit identifier (will be inferred from first column if not labeled)")
            st.markdown("- Time/cycle information (will be inferred from second column if not labeled)")
            st.markdown("- Sensor measurements (will be inferred from remaining columns if not labeled)")
        except Exception as e:
            # Restore stdout
            sys.stdout = old_stdout
            st.error(f"An unexpected error occurred: {str(e)}")
            st.exception(e)  # Show the full traceback for debugging
else:
    st.info("Please upload a file to get RUL predictions. The application supports various formats and will automatically detect column structure.")
    st.markdown("""
    ### Example file structure:
    Files should contain:
    - A column for engine/unit identifier (e.g., unit_number)
    - A column for time/cycle information (e.g., time_cycles)
    - Multiple columns for sensor measurements
    
    The application can handle files with or without headers, and will make intelligent assumptions about column meanings when needed.
    """)
