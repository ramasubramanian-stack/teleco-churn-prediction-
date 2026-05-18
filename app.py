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
    Batch predictions from uploaded CSV file
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Ensure all expected columns are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only feature columns
        input_df = df[feature_columns]
        
        # Make predictions
        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)
        
        # Add to dataframe
        results = df.copy()
        results['churn_probability'] = probabilities
        results['churn_prediction'] = predictions
        results['prediction_label'] = results['churn_prediction'].map({0: 'Unlikely', 1: 'Likely'})
        
        # Save results
        results.to_csv('predictions_results.csv', index=False)
        
        summary = {
            'total_customers': len(results),
            'predicted_churners': int(results['churn_prediction'].sum()),
            'churn_rate': float(results['churn_prediction'].mean()),
            'avg_churn_probability': float(results['churn_probability'].mean()),
            'file_saved': 'predictions_results.csv'
        }
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
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
