import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
            'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
            'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
            'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
            'sensor_21'
        ]
        self.operational_columns = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data"""
        try:
            # Select relevant columns
            features = data[self.feature_columns + self.operational_columns]
            
            # Scale the features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create sequences for LSTM input
            sequences = self._create_sequences(scaled_features)
            
            return sequences, scaled_features
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def _create_sequences(self, data: np.ndarray, sequence_length: int = 30) -> np.ndarray:
        """Create sequences for LSTM input"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
        
    def calculate_rul(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Remaining Useful Life"""
        max_cycles = data.groupby('unit')['time'].max()
        current_cycles = data.groupby('unit')['time'].last()
        rul = max_cycles - current_cycles
        return rul.values
        
    def get_feature_importance(self, model_type: str, data: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using SHAP values"""
        try:
            import shap
            
            # Load the model
            from utils.model_loader import ModelLoader
            model_loader = ModelLoader()
            model = model_loader.load_model(model_type)
            
            # Calculate SHAP values
            explainer = shap.DeepExplainer(model, data[:100])  # Use first 100 samples as background
            shap_values = explainer.shap_values(data)
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_importance = feature_importance.mean(axis=0)
            
            # Map to feature names
            feature_names = self.feature_columns + self.operational_columns
            importance_dict = dict(zip(feature_names, feature_importance))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
            
    def generate_synthetic_data(self, num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic turbofan degradation data"""
        try:
            # Generate base degradation pattern
            time = np.linspace(0, 100, num_samples)
            degradation = np.exp(-0.1 * time)
            
            # Generate sensor readings with noise
            sensor_data = {}
            for i in range(1, 22):
                noise = np.random.normal(0, 0.1, num_samples)
                sensor_data[f'sensor_{i}'] = degradation + noise
                
            # Generate operational settings
            op_settings = {
                'op_setting_1': np.random.uniform(0.5, 1.0, num_samples),
                'op_setting_2': np.random.uniform(0.3, 0.8, num_samples),
                'op_setting_3': np.random.uniform(0.2, 0.7, num_samples)
            }
            
            # Combine all data
            synthetic_data = pd.DataFrame({**sensor_data, **op_settings})
            synthetic_data['time'] = time
            synthetic_data['unit'] = 1  # All data belongs to one unit
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise 