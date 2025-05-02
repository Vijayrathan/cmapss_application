# Turbofan Engine RUL Prediction System

A comprehensive application for predicting Remaining Useful Life (RUL) of turbofan engines using various machine learning models.

## Features

1. **Real-time Prediction Dashboard**

   - Upload engine sensor logs or stream simulated test data
   - View predicted RUL per engine unit
   - Health degradation curves (LSTM outputs over time)
   - Confidence intervals using ensemble uncertainty

2. **Sensor Importance & Feature Diagnostics**

   - SHAP/LIME explainability for sensor contributions
   - Auto feature-scaling and correlation heatmaps

3. **Model Suite**

   - LSTM-based baseline model
   - Multiple model options: Transformer, Bi-LSTM, LSTM, CNN-LSTM, LightGBM, Multi-head attention LSTM
   - Comprehensive model evaluation metrics

4. **Retraining and Feedback Loop**

   - GUI for model retraining on new data
   - Label correction interface
   - Model drift detection and performance monitoring

5. **Synthetic Data Generator**
   - Generate synthetic turbofan degradation sequences
   - Support for Gaussian Process noise and rule-based patterns

## Project Structure

```
turbofan-rul/
├── backend/              # Flask backend
│   ├── api/             # API endpoints
│   ├── models/          # ML models
│   ├── utils/           # Utility functions
│   └── config/          # Configuration files
├── frontend/            # Angular frontend
│   ├── src/
│   └── assets/
├── data/                # Data storage
│   ├── raw/            # Raw data files
│   └── processed/      # Processed data files
└── tests/              # Test files
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up MongoDB:

   - Install MongoDB
   - Create a `.env` file with your MongoDB connection string

3. Start the backend:

   ```bash
   cd backend
   python app.py
   ```

4. Start the frontend:
   ```bash
   cd frontend
   ng serve
   ```

## API Documentation

The API documentation is available at `/api/docs` when running the backend server.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
