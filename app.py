import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and preprocessing
model = None
feature_columns = None
scaler = None
THRESHOLD = 0.30

def load_model():
    """Load pre-trained model and preprocessing objects"""
    global model, feature_columns, scaler
    try:
        if os.path.exists('models/xgb_model.pkl'):
            with open('models/xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
            logger.info("✓ Model loaded successfully")
        
        if os.path.exists('models/feature_columns.pkl'):
            with open('models/feature_columns.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
            logger.info("✓ Feature columns loaded")
                
        if os.path.exists('models/scaler.pkl'):
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✓ Scaler loaded")
    except Exception as e:
        logger.warning(f"Could not load saved model: {e}. Model will be trained on first request.")


@app.route('/')
def home():
    """Render the interactive dashboard UI"""
    return render_template('index.html')

'''
@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'Telco Customer Churn Prediction',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/predict': 'Make churn prediction for a customer',
            'GET /api/health': 'Health check',
            'POST /api/batch_predict': 'Batch predictions from CSV'
        }
    }), 200
'''

@app.route('/api/health', methods=['GET'])
def health():
    """Detailed health check"""
    health_status = {
        'model_loaded': model is not None,
        'threshold': THRESHOLD,
        'status': 'healthy' if model is not None else 'model_not_loaded'
    }
    return jsonify(health_status), 200 if model is not None else 503

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make churn prediction for a single customer
    
    Expected JSON format:
    {
        "tenure": 1,
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "gender": 0,  # 0=Female, 1=Male
        "Partner": 1,  # 0=No, 1=Yes
        "Dependents": 0,
        "PhoneService": 0,
        "PaperlessBilling": 0,
        "InternetService_Fiber optic": 0,
        ... (all encoded features)
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        
        # Create DataFrame with proper feature columns
        input_df = pd.DataFrame([data])
        
        # Ensure all expected columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only the feature columns used during training
        input_df = input_df[feature_columns]
        
        # Make prediction
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability >= THRESHOLD else 0
        
        response = {
            'churn_probability': float(probability),
            'churn_prediction': int(prediction),
            'threshold_used': THRESHOLD,
            'prediction_label': 'Likely to Churn' if prediction == 1 else 'Unlikely to Churn',
            'recommendation': 'Consider retention campaign' if prediction == 1 else 'No action needed'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Robust batch processing that handles partial, raw, or misaligned test CSV files.
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        if df.empty:
            return jsonify({'error': 'The uploaded CSV file contains no data rows.'}), 400
        
        # Save a clean copy of whatever the user uploaded to append our results onto
        results = df.copy()
        
        # Standardize column headers by stripping out accidental trailing spaces
        df.columns = df.columns.str.strip()
        
        # 1. Safely handle binary string columns if they exist
        binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
            else:
                df[col] = 0
        
        # 2. Safely handle multi-category one-hot encoding
        multi_cat = [
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaymentMethod"
        ]
        existing_multi = [c for c in multi_cat if c in df.columns]
        if existing_multi:
            df = pd.get_dummies(df, columns=existing_multi, drop_first=True)
        
        # 3. Safe extraction for TotalCharges to completely avoid a KeyError
        if 'TotalCharges' in df.columns:
            if df['TotalCharges'].dtype == 'object':
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            fill_value = df['TotalCharges'].median()
            if pd.isna(fill_value):
                fill_value = 0
            df['TotalCharges'] = df['TotalCharges'].fillna(fill_value)
        else:
            df['TotalCharges'] = 0
            
        # 4. Fill absolute missing feature matrix layers with 0
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Force the column sequence to match the model training layout exactly
        input_df = df[feature_columns]
        
        # Run Matrix Inference
        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)
        
        # Add output metrics onto the user's uploaded spreadsheet copy
        results['churn_probability'] = probabilities
        results['churn_prediction'] = predictions
        results['prediction_label'] = results['churn_prediction'].map({0: 'Unlikely to Churn', 1: 'Likely to Churn'})
        
        csv_string = results.to_csv(index=False)
        
        summary = {
            'total_customers': len(results),
            'predicted_churners': int(predictions.sum()),
            'churn_rate': float(predictions.mean()),
            'avg_churn_probability': float(probabilities.mean()),
            'csv_string': csv_string
        }
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Batch processing exception raised: {str(e)}")
        return jsonify({'error': str(e)}), 400
@app.route('/api/features', methods=['GET'])
def get_features():
    """Return list of expected features"""
    if feature_columns is None:
        return jsonify({'error': 'Features not loaded'}), 503
    
    return jsonify({
        'feature_count': len(feature_columns),
        'features': sorted(feature_columns)
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'available_endpoints': [
        'GET /',
        'GET /api/health',
        'POST /api/predict',
        'POST /api/batch_predict',
        'GET /api/features'
    ]}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

load_model()
 
if __name__ == '__main__':
    # Load model on startup
   
    
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
