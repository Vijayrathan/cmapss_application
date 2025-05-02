import torch
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_type: str) -> Optional[Any]:
        """Load a specific model type"""
        if model_type in self.models:
            return self.models[model_type]
            
        model_path = os.path.join(self.model_dir, f"{model_type.lower()}_rul_model.pt")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
            
        try:
            # Load the model based on type
            if model_type == "LSTM":
                from models.lstm import LSTMModel
                model = LSTMModel()
            elif model_type == "Bi-LSTM":
                from models.bilstm import BiLSTMModel
                model = BiLSTMModel()
            elif model_type == "Transformer":
                from models.transformer import TransformerModel
                model = TransformerModel()
            elif model_type == "CNN-LSTM":
                from models.cnnlstm import CNNLSTMModel
                model = CNNLSTMModel()
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
                
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models[model_type] = model
            logger.info(f"Successfully loaded {model_type} model")
            return model
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            return None
            
    def get_available_models(self) -> list:
        """Get list of available model files"""
        available_models = []
        for file in os.listdir(self.model_dir):
            if file.endswith("_rul_model.pt"):
                model_type = file.split("_")[0].upper()
                available_models.append(model_type)
        return available_models
        
    def predict(self, model_type: str, data: torch.Tensor) -> torch.Tensor:
        """Make prediction using specified model"""
        model = self.load_model(model_type)
        if model is None:
            raise ValueError(f"Model {model_type} not available")
            
        with torch.no_grad():
            data = data.to(self.device)
            prediction = model(data)
            return prediction.cpu() 