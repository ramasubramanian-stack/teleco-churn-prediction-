"""
Training script for Telco Customer Churn prediction
This script trains the XGBoost model and saves it for deployment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.30
DATA_PATH = 'data/Telco-Customer-Churn.csv'
MODEL_DIR = 'models'

def ensure_model_dir():
    """Create models directory if it doesn't exist"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"✓ Model directory ready: {MODEL_DIR}")

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the Telco customer churn dataset
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Binary encoding for 2-category features
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    
    df[binary_cols] = df[binary_cols].replace({
        'Yes': 1, 'No': 0,
        'Male': 1, 'Female': 0
    }).astype('int64')
    
    logger.info("✓ Binary encoding completed")
    
    # One-hot encoding for multi-category features
    multi_cat = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    
    df = pd.get_dummies(df, columns=multi_cat, drop_first=True)
    logger.info("✓ One-hot encoding completed")
    
    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop unnecessary columns
    df = df.drop("customerID", axis=1)
    
    # Convert bool to int
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # Handle missing values in TotalCharges
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    logger.info("✓ Data preprocessing completed")
    logger.info(f"Final dataset shape: {df.shape}")
    
    # Check churn distribution
    churn_dist = df['Churn'].value_counts()
    logger.info(f"Churn distribution:\n{churn_dist}")
    logger.info(f"Churn rate: {df['Churn'].mean():.2%}")
    
    return df

def train_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier with optimized hyperparameters
    """
    logger.info("Training XGBoost model...")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Class weight ratio: {scale_pos_weight:.2f}")
    
    # Optimized hyperparameters (from Optuna tuning)
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbose=0
    )
    
    xgb.fit(X_train, y_train)
    logger.info("✓ Model training completed")
    
    # Evaluate
    proba = xgb.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL PERFORMANCE (Threshold = 0.30)")
    logger.info("="*60)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Summary metrics
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"F1-Score: {f1:.3f}")
    logger.info("="*60 + "\n")
    
    return xgb

def save_model_and_artifacts(model, feature_columns):
    """
    Save model and preprocessing artifacts
    """
    ensure_model_dir()
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✓ Model saved: {model_path}")
    
    # Save feature columns
    features_path = os.path.join(MODEL_DIR, 'feature_columns.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    logger.info(f"✓ Feature columns saved: {features_path}")
    
    # Save configuration
    config = {
        'threshold': THRESHOLD,
        'feature_count': len(feature_columns),
        'random_state': RANDOM_STATE
    }
    config_path = os.path.join(MODEL_DIR, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    logger.info(f"✓ Configuration saved: {config_path}")

def main():
    """
    Main training pipeline
    """
    logger.info("="*60)
    logger.info("TELCO CUSTOMER CHURN - MODEL TRAINING")
    logger.info("="*60 + "\n")
    
    # 1. Load and preprocess
    df = load_and_preprocess_data(DATA_PATH)
    
    # 2. Prepare features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    logger.info(f"\nFeatures shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}\n")
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}\n")
    
    # 4. Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # 5. Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    logger.info(feature_importance.head(10).to_string(index=False))
    
    # 6. Save model and artifacts
    save_model_and_artifacts(model, X.columns.tolist())
    
    logger.info("\n" + "="*60)
    logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info("\nModel ready for deployment!")
    logger.info(f"Model files saved in: {MODEL_DIR}/")
    
    return model, X.columns.tolist()

if __name__ == '__main__':
    main()
