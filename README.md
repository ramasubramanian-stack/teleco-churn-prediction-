# Telco Customer Churn Prediction API

A machine learning API for predicting customer churn in telecommunications using XGBoost. Deployed on Render.

## 🎯 Project Overview

This project analyzes the Telco Customer Churn dataset and deploys a production-ready ML model that predicts which customers are likely to leave the service. The model is accessible via a REST API.

### Key Features
- **Model:** XGBoost with 82% recall (catches most churners)
- **Predictions:** Single customer or batch predictions
- **API:** Flask-based REST API
- **Deployment:** Render (free tier compatible)
- **Threshold:** Optimized at 0.30 for maximum churn detection

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Recall | 82% |
| Precision | 60% |
| F1-Score | 70% |
| Threshold | 0.30 |

**Why recall matters:** Missing a churner (false negative) is more costly than incorrectly flagging a loyal customer (false positive).

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/telco-churn-api.git
cd telco-churn-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
mkdir -p data
# Place Telco-Customer-Churn.csv in data/ folder

# 4. Train model
python train.py

# 5. Run API
python app.py

# 6. Test
curl http://localhost:8000/
```

### Deploy to Render
1. Push code to GitHub
2. Connect repo to Render
3. Render auto-deploys using `render.yaml`
4. Access via `https://your-app.onrender.com`

**See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions**

## 📡 API Usage

### Health Check
```bash
curl https://your-app.onrender.com/
```

### Predict Single Customer
```bash
curl -X POST https://your-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "MonthlyCharges": 65.50,
    "TotalCharges": 1570.00,
    "gender": 1,
    "Partner": 1,
    ... (add all features)
  }'
```

### Batch Predictions
```bash
curl -X POST https://your-app.onrender.com/api/batch_predict \
  -F "file=@customers.csv"
```

### Get Feature List
```bash
curl https://your-app.onrender.com/api/features
```

## 📁 Repository Structure

```
.
├── app.py                      # Flask API
├── train.py                    # Training pipeline
├── requirements.txt            # Dependencies
├── render.yaml                 # Render config
├── README.md                   # This file
├── DEPLOYMENT_GUIDE.md         # Full deployment docs
├── .gitignore                  # Git ignore rules
├── data/                       # Dataset (create locally)
│   └── Telco-Customer-Churn.csv
├── models/                     # Trained models (auto-created)
│   ├── xgb_model.pkl
│   ├── feature_columns.pkl
│   └── config.pkl
└── notebooks/
    └── EDA.ipynb              # Exploratory analysis
```

## 🔧 Configuration

### Model Hyperparameters
Edit in `train.py`:
```python
THRESHOLD = 0.30              # Prediction threshold
TEST_SIZE = 0.2               # Train-test split
RANDOM_STATE = 42             # Reproducibility
```

### Environment Variables
```bash
FLASK_ENV=production          # Run mode
PORT=8000                     # Server port (auto-set on Render)
PYTHON_VERSION=3.11           # Python version
```

## 📊 Data Overview

**Dataset:** Telco Customer Churn
- **Samples:** ~7,000 customers
- **Features:** 20 (after encoding)
- **Target:** Churn (Yes/No)
- **Churn Rate:** ~27%

### Feature Groups
1. **Demographics:** gender, Partner, Dependents
2. **Services:** PhoneService, InternetService, OnlineSecurity, etc.
3. **Billing:** MonthlyCharges, TotalCharges, Contract, PaymentMethod
4. **Account:** tenure, Churn

### Encoding Strategy
- **Binary:** Yes/No → 1/0, Male/Female → 1/0
- **Multi-class:** One-hot encoding (drop first to avoid collinearity)

## 🤖 Model Selection

Tested three models:
1. **RandomForest:** 71.7% recall, slower training
2. **LightGBM:** 82% recall, fast training
3. **XGBoost:** 82% recall, 3x faster than LightGBM ✓ **Selected**

## 📈 Training Pipeline

```
1. Load & Preprocess Data
   ↓
2. Binary Encoding (2-category features)
   ↓
3. One-Hot Encoding (multi-category features)
   ↓
4. Clean Missing Values
   ↓
5. Train-Test Split (80-20)
   ↓
6. Train XGBoost Model
   ↓
7. Threshold Tuning (optimize for recall)
   ↓
8. Save Model & Artifacts
   ↓
9. Ready for Deployment
```

## 🧪 Testing

### Unit Test Example
```python
import requests

api_url = "http://localhost:8000/api/predict"
sample_customer = {
    "tenure": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "gender": 0,
    # ... add all features
}

response = requests.post(api_url, json=sample_customer)
print(response.json())
# Expected: churn_probability, churn_prediction, recommendation
```

## 🔐 Security & Production

### Current Limitations
- No authentication (add API keys in production)
- No rate limiting
- No input validation (add Marshmallow schemas)

### Recommended Improvements
```python
# Add to production:
from flask_limiter import Limiter
from flask_jwt_extended import JWTManager
from marshmallow import Schema, fields, validate
```

## 📚 Feature Importance

Top features driving churn predictions:
1. **Tenure** (-0.35 correlation) - Longer tenure = less churn
2. **Contract_Two year** (-0.30) - Long contracts = loyalty
3. **InternetService_Fiber optic** (+0.31) - Fiber users churn more
4. **PaymentMethod_Electronic check** (+0.30) - Check payment correlates with churn

## 🚀 Deployment Checklist

- [ ] Data file added to `data/` folder
- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] GitHub repo connected to Render
- [ ] `render.yaml` configured
- [ ] Deployment logs verified
- [ ] API endpoints tested
- [ ] Custom domain added (optional)

## 📞 Support & Resources

- **Render Docs:** https://render.com/docs
- **Flask Documentation:** https://flask.palletsprojects.com/
- **XGBoost Docs:** https://xgboost.readthedocs.io/
- **Scikit-learn:** https://scikit-learn.org/

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

**Status:** Production Ready ✓  
**Last Updated:** January 2024  
**Maintained By:** Your Name
