import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class DriftDetector:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client.turbofan_rul
        self.performance_collection = self.db.model_performance
        
    def log_performance(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Log model performance metrics"""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
                
            performance_data = {
                "model_type": self.model_type,
                "timestamp": timestamp,
                "metrics": metrics
            }
            
            self.performance_collection.insert_one(performance_data)
            logger.info(f"Logged performance metrics for {self.model_type}")
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {str(e)}")
            
    def detect_drift(self, window_size: int = 30) -> Dict[str, float]:
        """Detect model drift using recent performance metrics"""
        try:
            # Get recent performance metrics
            recent_metrics = list(self.performance_collection
                                .find({"model_type": self.model_type})
                                .sort("timestamp", -1)
                                .limit(window_size))
            
            if not recent_metrics:
                return {"drift_detected": False, "confidence": 0.0}
                
            # Calculate baseline metrics (first half of window)
            baseline_size = window_size // 2
            baseline_metrics = recent_metrics[baseline_size:]
            current_metrics = recent_metrics[:baseline_size]
            
            # Calculate drift scores for each metric
            drift_scores = {}
            for metric in baseline_metrics[0]["metrics"].keys():
                baseline_values = [m["metrics"][metric] for m in baseline_metrics]
                current_values = [m["metrics"][metric] for m in current_metrics]
                
                # Calculate drift using Kolmogorov-Smirnov test
                from scipy import stats
                drift_score = stats.ks_2samp(baseline_values, current_values)[0]
                drift_scores[metric] = drift_score
                
            # Calculate overall drift confidence
            max_drift = max(drift_scores.values())
            drift_detected = max_drift > 0.2  # Threshold for drift detection
            
            return {
                "drift_detected": drift_detected,
                "confidence": max_drift,
                "metric_scores": drift_scores
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            return {"drift_detected": False, "confidence": 0.0}
            
    def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get performance history for the specified number of days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            history = list(self.performance_collection
                         .find({
                             "model_type": self.model_type,
                             "timestamp": {"$gte": cutoff_date}
                         })
                         .sort("timestamp", 1))
                         
            return history
            
        except Exception as e:
            logger.error(f"Error getting performance history: {str(e)}")
            return []
            
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {} 