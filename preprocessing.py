import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from scipy import stats
import warnings
import os

def load_and_preprocess(file_path, seq_len=30):
    """
    Preprocesses data from various file formats (csv, txt) with or without headers.
    Handles automatic column detection and naming.
    """
    # Suppress warnings during preprocessing
    warnings.filterwarnings('ignore')
    
    # Detect file extension
    file_ext = os.path.splitext(file_path)[1].lower() if isinstance(file_path, str) else ''
    
    # Try to detect if file has headers
    try:
        # Check first few lines for header detection
        if isinstance(file_path, str):
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                has_header = not all(c.isdigit() or c in '.,- \t' for c in first_line)
        else:
            # For uploaded files through streamlit
            file_path.seek(0)
            first_line = file_path.readline().decode().strip()
            has_header = not all(c.isdigit() or c in '.,- \t' for c in first_line)
            file_path.seek(0)  # Reset file pointer
    except:
        # Default to assuming headers
        has_header = True
    
    # Special handling for turbofan engine dataset format (FD00x.txt files)
    is_turbofan_format = False
    try:
        if isinstance(file_path, str):
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # Check if it matches the specific format of NASA Turbofan dataset
                if 'FD00' in file_path or (len(first_line.split()) > 5 and first_line.split()[0].isdigit() and first_line.split()[1].isdigit()):
                    is_turbofan_format = True
                    print("Detected NASA Turbofan dataset format")
        else:
            file_path.seek(0)
            first_line = file_path.readline().decode().strip()
            file_name = getattr(file_path, 'name', '')
            if 'FD00' in file_name or (len(first_line.split()) > 5 and first_line.split()[0].isdigit() and first_line.split()[1].isdigit()):
                is_turbofan_format = True
                print("Detected NASA Turbofan dataset format")
            file_path.seek(0)
    except:
        pass
    
    # Load the data based on file type and header presence
    try:
        if is_turbofan_format:
            # For NASA Turbofan dataset format
            if isinstance(file_path, str):
                # Read the file line by line and parse manually
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        # Make sure we have at least the minimum required values
                        if len(values) >= 2:
                            # Convert all values to float where possible
                            processed_values = []
                            for val in values:
                                try:
                                    processed_values.append(float(val))
                                except:
                                    processed_values.append(val)
                            data.append(processed_values)
            else:
                # For streamlit uploaded files
                data = []
                file_path.seek(0)
                for line in file_path:
                    values = line.decode().strip().split()
                    if len(values) >= 2:
                        processed_values = []
                        for val in values:
                            try:
                                processed_values.append(float(val))
                            except:
                                processed_values.append(val)
                        data.append(processed_values)
            
            # Create standard column names for the NASA dataset
            num_columns = max(len(row) for row in data) if data else 0
            column_names = []
            for i in range(num_columns):
                if i == 0:
                    column_names.append('unit_number')
                elif i == 1:
                    column_names.append('time_cycles')
                elif i < 5:
                    column_names.append(f'setting_{i-1}')
                else:
                    column_names.append(f's_{i-4}')
            
            # Create dataframe with fixed column names
            if data and column_names:
                df = pd.DataFrame(data, columns=column_names)
                print(f"Processed NASA Turbofan data with {df.shape[1]} columns")
            else:
                raise ValueError("No valid data found in the file")
            
        elif file_ext == '.csv' or file_ext == '':
            df = pd.read_csv(file_path, header=0 if has_header else None)
        elif file_ext == '.txt':
            # Try different delimiters for txt files
            for delimiter in [' ', ',', '\t', ';']:
                try:
                    df = pd.read_csv(file_path, header=0 if has_header else None, delimiter=delimiter)
                    # If we got reasonable number of columns, break
                    if df.shape[1] > 1:
                        break
                except:
                    continue
        else:
            # Try to guess the format
            df = pd.read_csv(file_path, header=0 if has_header else None)
    except Exception as e:
        raise ValueError(f"Could not load file: {str(e)}")
    
    # If no headers were found, create standard column names
    if not has_header:
        # Rename columns to standard format
        new_columns = []
        for i in range(df.shape[1]):
            if i == 0:
                new_columns.append('unit_number')
            elif i == 1:
                new_columns.append('time_cycles')
            elif i < 5:  # Assume first few columns might be settings
                new_columns.append(f'setting_{i-1}')
            else:
                # Assume the rest are sensor readings
                new_columns.append(f's_{i-4}')
        df.columns = new_columns
        print(f"No headers detected. Created standard columns: {new_columns}")
    
    # Check if we have the minimum required columns or try to infer them
    if 'unit_number' not in df.columns and 'time_cycles' not in df.columns:
        # Try to infer columns based on patterns
        if df.shape[1] >= 2:
            # Assume first column is unit_number, second is time_cycles
            first_col = df.columns[0]
            second_col = df.columns[1]
            df = df.rename(columns={first_col: 'unit_number', second_col: 'time_cycles'})
            print(f"Renamed columns {first_col} → unit_number, {second_col} → time_cycles")
    
    # Try to infer sensor columns if none are found
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    if not sensor_cols and df.shape[1] > 2:
        # Assume columns after the first two are sensor readings
        for i in range(2, min(df.shape[1], 26)):  # Limit to reasonable number
            col_name = df.columns[i]
            new_name = f's_{i-1}'
            df = df.rename(columns={col_name: new_name})
        sensor_cols = [c for c in df.columns if c.startswith('s_')]
        print(f"Inferred sensor columns: {sensor_cols}")
    
    # Safely convert columns to numeric where possible
    for col in df.columns:
        try:
            # Check if the column is already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Try to convert to numeric, filling non-convertible values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert column {col} to numeric: {str(e)}")
    
    # Ensure we have unit_number and time_cycles as numeric
    if 'unit_number' in df.columns:
        try:
            df['unit_number'] = df['unit_number'].astype(float).fillna(1).astype(int)
        except:
            print("Warning: Could not convert unit_number to numeric, using default values")
            df['unit_number'] = range(1, len(df) + 1)
    else:
        # Create a default unit_number column
        df['unit_number'] = range(1, len(df) + 1)
        print("Added default unit_number = 1")
    
    if 'time_cycles' in df.columns:
        try:
            df['time_cycles'] = df['time_cycles'].astype(float).fillna(df.index + 1).astype(int)
        except:
            print("Warning: Could not convert time_cycles to numeric, using row index")
            df['time_cycles'] = df.index + 1
    else:
        # Use the row index as time_cycles
        df['time_cycles'] = df.index + 1
        print("Created time_cycles from row index")
    
    # Print number of unique engines
    num_engines = df['unit_number'].nunique()
    print(f"File contains {num_engines} unique engines")
    
    # Check for RUL column, if not exist create it
    if 'RUL' not in df.columns:
        # Create RUL by grouping by unit_number and calculating max(time_cycles) - time_cycles
        df = df.sort_values(['unit_number', 'time_cycles'])
        max_cycles = df.groupby('unit_number')['time_cycles'].transform('max')
        df['RUL'] = max_cycles - df['time_cycles']
        print("Created RUL from time_cycles data")
    
    # ====== FEATURE ENGINEERING ======
    # Time-based features
    df['time_since_start'] = df.groupby('unit_number')['time_cycles'].transform(lambda x: x - x.min())
    df['time_to_failure'] = df.groupby('unit_number')['time_cycles'].transform('max') - df['time_cycles']
    df['cycle_normalized'] = df['time_cycles'] / (df.groupby('unit_number')['time_cycles'].transform('max') + 1e-8)
    
    # Select sensor and setting columns
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    setting_cols = [c for c in df.columns if c.startswith('setting_')]
    
    # If no settings were found, try to infer them
    if not setting_cols and df.shape[1] > 2:
        potential_settings = [c for c in df.columns if c not in sensor_cols + ['unit_number', 'time_cycles', 'RUL', 
                                                                             'time_since_start', 'time_to_failure', 
                                                                             'cycle_normalized']]
        # Take up to 3 columns as settings
        for i, col in enumerate(potential_settings[:3]):
            new_name = f'setting_{i+1}'
            df = df.rename(columns={col: new_name})
        setting_cols = [c for c in df.columns if c.startswith('setting_')]
        if setting_cols:
            print(f"Inferred setting columns: {setting_cols}")
    
    # Replace any infinite values with NaN first
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN values with median for each engine
    for col in sensor_cols + setting_cols:
        df[col] = df.groupby('unit_number')[col].transform(lambda x: x.fillna(x.median()))
    
    # If still any NaNs, fill with global median
    df = df.fillna(df.median())
    
    # ====== OUTLIER HANDLING ======
    # Handle outliers in sensor readings using winsorization with safe handling
    for col in sensor_cols:
        try:
            df[col] = winsorize_column(df[col])
        except:
            print(f"Warning: Could not winsorize column {col}, using original values")
    
    # ====== ADDITIONAL FEATURES ======
    # Add derivative features (rate of change) for sensors
    for col in sensor_cols:
        # First order difference
        df[f'{col}_diff'] = df.groupby('unit_number')[col].diff().fillna(0)
        
        # Second order difference (with safe handling)
        try:
            df[f'{col}_diff2'] = df.groupby('unit_number')[f'{col}_diff'].diff().fillna(0)
        except:
            df[f'{col}_diff2'] = 0
    
    # Add moving averages and standard deviations (safely)
    window_sizes = [3, 5]  # Reduced from [3, 5, 7]
    for col in sensor_cols:
        for window in window_sizes:
            # Skip if we don't have enough data
            if len(df) >= window:
                try:
                    # Moving average
                    df[f'{col}_ma{window}'] = df.groupby('unit_number')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean())
                    
                    # Moving standard deviation (with safe handling for too few points)
                    df[f'{col}_std{window}'] = df.groupby('unit_number')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
                except:
                    # If errors, use safer approaches
                    df[f'{col}_ma{window}'] = df[col]
                    df[f'{col}_std{window}'] = 0
    
    # Create sensor interaction terms (for select sensors only, with safe division)
    # Use only first 3 sensors to reduce feature count and potential errors
    important_sensors = sensor_cols[:3] if len(sensor_cols) >= 3 else sensor_cols
    for i, s1 in enumerate(important_sensors):
        for s2 in important_sensors[i+1:]:
            # Safe division (adding small epsilon to denominator)
            df[f'{s1}_{s2}_ratio'] = df[s1] / (df[s2] + 1e-8)
            df[f'{s1}_{s2}_sum'] = df[s1] + df[s2]
    
    # Clip extreme values (another safety measure)
    for col in df.columns:
        if col not in ['unit_number', 'time_cycles', 'RUL']:
            try:
                df[col] = df[col].clip(lower=-1e9, upper=1e9)
            except:
                pass
    
    # Fill any NaN values that might have been created
    df = df.fillna(0)
    
    # Replace any infinite values that might have been created
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Get all engineered features
    derivative_cols = [col for col in df.columns if '_diff' in col]
    moving_avg_cols = [col for col in df.columns if '_ma' in col or '_std' in col]
    interaction_cols = [col for col in df.columns if '_ratio' in col or '_sum' in col]
    other_feature_cols = ['time_since_start', 'time_to_failure', 'cycle_normalized']
    
    # Combine all features
    all_feature_cols = sensor_cols + setting_cols + derivative_cols + moving_avg_cols + interaction_cols + other_feature_cols
    
    # ====== SCALING ======
    # Apply scaling safely
    try:
        # Check for NaN or Inf before scaling
        if df[all_feature_cols].isna().any().any() or np.isinf(df[all_feature_cols].values).any():
            print("Warning: Data contains NaN or Inf values before scaling")
            # Additional cleanup if needed
            df[all_feature_cols] = df[all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale features using RobustScaler (less sensitive to outliers)
        scaler = RobustScaler()
        df[all_feature_cols] = scaler.fit_transform(df[all_feature_cols])
    except Exception as e:
        print(f"Warning: Scaling error: {e}")
        print("Falling back to basic scaling approach")
        
        # Fallback to simple min-max scaling
        for col in all_feature_cols:
            if df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Final NaN check and replacement
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    # ====== SEQUENCE GENERATION ======
    sequences, labels, ids = [], [], []
    for eid, group in df.groupby('unit_number'):
        data = group[all_feature_cols].values
        rul_values = group['RUL'].values
        
        if len(data) >= seq_len:
            for i in range(len(group) - seq_len + 1):
                seq = data[i:i + seq_len]
                label = rul_values[i + seq_len - 1]
                
                # Skip sequences with NaN or Inf
                if np.isnan(seq).any() or np.isinf(seq).any() or np.isnan(label) or np.isinf(label):
                    print(f"Warning: Skipping sequence with NaN/Inf for engine {eid}")
                    continue
                    
                sequences.append(seq)
                labels.append(label)
                ids.append(eid)
    
    if not sequences:
        raise ValueError("No valid sequences found. Make sure the data has enough rows per unit_number")
    
    print(f"Created {len(sequences)} valid sequences from {len(np.unique(ids))} engines")
    
    # Reset warnings
    warnings.resetwarnings()
    
    return np.array(sequences), np.array(ids), np.array(labels)

def winsorize_column(column, limits=(0.05, 0.05)):
    """
    Winsorize a column to limit the effect of outliers
    """
    # Make a copy to avoid modifying the original
    col_no_na = column.dropna()
    
    # Only winsorize if we have enough values
    if len(col_no_na) > 10:
        winsorized = pd.Series(stats.mstats.winsorize(col_no_na, limits=limits), index=col_no_na.index)
        # Merge back with original to preserve NaN positions
        result = column.copy()
        result.loc[winsorized.index] = winsorized
        return result
    else:
        return column
