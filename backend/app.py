# backend/app.py - OPTIMIZED LOGISTIC REGRESSION WITH CORRECT IMPORTS
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle
import uuid
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE, handle if not available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✅ SMOTE available for handling imbalanced data")
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ SMOTE not available. Install with: pip install imbalanced-learn")
    print("   Continuing without SMOTE...")

app = FastAPI(title="AI Churn Analysis API", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_NAME = "churn_analysis.db"

def init_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id TEXT PRIMARY KEY,
        age_group TEXT, platforms TEXT, primary_platform TEXT,
        subscription_duration TEXT, subscription_plan TEXT, monthly_spend TEXT,
        usage_frequency TEXT, nps_score INTEGER, content_satisfaction INTEGER,
        value_perception TEXT, churn_likelihood TEXT, switching_behavior TEXT,
        switching_reason TEXT, account_sharing TEXT, competitor_awareness TEXT,
        bundling_preference TEXT, payment_method TEXT, suggestions TEXT,
        created_at TIMESTAMP, churn_probability REAL, risk_level TEXT,
        prediction_confidence REAL
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY, customer_id TEXT, churn_probability REAL,
        risk_level TEXT, key_factors TEXT, recommendations TEXT,
        confidence REAL, created_at TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers (id)
    )
    ''')
    conn.commit()
    conn.close()

init_database()

class CustomerData(BaseModel):
    age_group: str
    platforms: Optional[List[str]] = []
    primary_platform: str
    subscription_duration: str
    subscription_plan: str
    monthly_spend: str
    usage_frequency: str
    nps_score: int
    content_satisfaction: int
    value_perception: str
    churn_likelihood: str
    switching_behavior: str
    switching_reason: Optional[str] = None
    account_sharing: str
    competitor_awareness: str
    bundling_preference: str
    payment_method: str
    suggestions: Optional[str] = None

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.training_metrics = {}
        self.threshold = 0.35
        
    def load_and_preprocess_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        standardized_df = pd.DataFrame()
        
        column_mapping = {
            'Which age group do you belong to?': 'age_group',
            'Which OTT platform(s) are you currently subscribed to?': 'platforms',
            'Which is your PRIMARY OTT subscription? (The one you use most)': 'primary_platform',
            'How long have you been continuously subscribed to your primary platform?': 'subscription_duration',
            'What subscription plan are you currently on?': 'subscription_plan',
            'Approximately how much do you spend MONTHLY on your primary subscription?': 'monthly_spend',
            'How often do you actually use your primary subscription?': 'usage_frequency',
            'On a scale of 0-10, how likely are you to recommend your primary platform to a friend?': 'nps_score',
            'How satisfied are you with the CONTENT on your primary platform?': 'content_satisfaction',
            'Do you feel you\'re getting good value for what you pay?': 'value_perception',
            'How likely are you to cancel your primary OTT subscription in the next 3 months?  ': 'churn_likelihood',
            'Have you switched from one OTT service to another in the last 6 months?': 'switching_behavior',
            'What is the main reason for switching or cancelling the subscription?': 'switching_reason',
            'Are you sharing your account with people outside your immediate household?': 'account_sharing',
            'Are you aware of competing platforms offering better value?': 'competitor_awareness',
            'Would you prefer bundled subscriptions (multiple platforms together) at a discount?': 'bundling_preference',
            'How do you pay for your subscription?': 'payment_method',
            'Any suggestions about your streaming experience?': 'suggestions'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                standardized_df[new_name] = df[old_name]
        
        if 'platforms' in standardized_df.columns:
            standardized_df['platforms'] = standardized_df['platforms'].apply(
                lambda x: str(x).split(',') if pd.notna(x) else []
            )
        
        if 'nps_score' in standardized_df.columns:
            standardized_df['nps_score'] = pd.to_numeric(standardized_df['nps_score'], errors='coerce').fillna(5).astype(int)
        
        if 'content_satisfaction' in standardized_df.columns:
            standardized_df['content_satisfaction'] = pd.to_numeric(standardized_df['content_satisfaction'], errors='coerce').fillna(3).astype(int)
        
        for col in standardized_df.columns:
            if standardized_df[col].dtype == 'object':
                standardized_df[col] = standardized_df[col].fillna('').astype(str).str.strip()
        
        return standardized_df
    
    def prepare_features(self, df, fit_encoders=True):
        df_processed = df.copy()
        
        categorical_cols = {
            'age_group': ['Under 18', '18-35', '35+'],
            'primary_platform': ['Netflix', 'Amazon Prime Video', 'Disney+ Hotstar', 'YouTube Premium'],
            'subscription_duration': ['Less than 1 month', '3-6 months', '6-12 months', '1-2 years', 'More than 2 years'],
            'subscription_plan': ['Basic', 'Standard', 'Premium'],
            'monthly_spend': ['Less than ₹200', '₹200 - ₹500', '₹501 - ₹1000', 'More than ₹1000'],
            'usage_frequency': ['Rarely', 'Often', 'Daily'],
            'value_perception': ['Poor value', 'Average', 'Good value', 'Excellent value'],
            'switching_behavior': ['No changes', 'Yes added new one', 'Yes switched completely'],
            'account_sharing': ['No only I use it', 'Yes with friends', 'Yes with family'],
            'competitor_awareness': ['No satisfied with current', 'Yes but not seriously', 'Yes actively considering'],
            'bundling_preference': ['No prefer separate', 'Maybe depends on price', 'Yes definitely'],
            'payment_method': ['Monthly auto-renewal', 'Annual payment', 'Pay as you go', 'Family member pays']
        }
        
        for col, categories in categorical_cols.items():
            if col in df_processed.columns:
                if fit_encoders and col not in self.label_encoders:
                    le = LabelEncoder()
                    le.fit(categories)
                    self.label_encoders[col] = le
                
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df_processed[f'{col}_encoded'] = df_processed[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
        
        def is_churn(value):
            value = str(value).lower()
            return 1 if 'very likely' in value or '10' in value else 0
        
        if 'churn_likelihood' in df_processed.columns:
            df_processed['churn'] = df_processed['churn_likelihood'].apply(is_churn)
        
        return df_processed
    
    def train_model(self, csv_path=None):
        """Train optimized Logistic Regression"""
        try:
            if csv_path:
                df = self.load_and_preprocess_csv(csv_path)
            else:
                if os.path.exists('OTT Platform Survey.csv'):
                    df = self.load_and_preprocess_csv('OTT Platform Survey.csv')
                else:
                    print("No CSV file found")
                    return False
            
            if df is None or len(df) < 10:
                print(f"Insufficient data: {len(df) if df is not None else 0} rows")
                return False
            
            print(f"\n{'='*60}")
            print(f"OPTIMIZED LOGISTIC REGRESSION TRAINING")
            print(f"{'='*60}")
            
            df_processed = self.prepare_features(df, fit_encoders=True)
            
            if 'churn' not in df_processed.columns:
                print("❌ Could not find churn column!")
                return False
            
            churn_count = df_processed['churn'].sum()
            non_churn_count = len(df_processed) - churn_count
            
            print(f"\nOriginal Class Distribution:")
            print(f"  - Churn: {churn_count} ({churn_count/len(df_processed)*100:.1f}%)")
            print(f"  - Non-Churn: {non_churn_count} ({non_churn_count/len(df_processed)*100:.1f}%)")
            
            self.feature_columns = [col for col in df_processed.columns if col.endswith('_encoded')]
            
            if not self.feature_columns:
                print("❌ No feature columns found!")
                return False
            
            X = df_processed[self.feature_columns].values
            y = df_processed['churn'].values
            X = np.nan_to_num(X)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"\nTraining samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Training churn rate: {y_train.mean()*100:.1f}%")
            
            # Apply SMOTE if available and needed
            if SMOTE_AVAILABLE and churn_count >= 3:
                print("\n⚖️ Applying SMOTE to balance classes...")
                smote = SMOTE(random_state=42, sampling_strategy=0.4)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE - Training samples: {len(X_train)}")
                print(f"After SMOTE - Churn rate: {y_train.mean()*100:.1f}%")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train Logistic Regression with class weights
            print("\n🔧 Training Logistic Regression...")
            self.model = LogisticRegression(
                C=0.5,  # Stronger regularization
                class_weight={0: 1, 1: 2.5},  # Less weight on churn class to reduce false positives
                random_state=42,
                max_iter=1000,
                solver='lbfgs'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Find optimal threshold
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            
            best_threshold = 0.55
            best_f1 = 0
            for threshold in np.arange(0.2, 0.65, 0.05):
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.threshold = best_threshold
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            auc = 0.5
            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, y_pred_proba)
            
            self.training_metrics = {
                'algorithm': 'Logistic Regression',
                'smote_used': SMOTE_AVAILABLE and churn_count >= 3,
                'threshold': self.threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'churn_count': int(churn_count)
            }
            
            print(f"\n{'='*60}")
            print(f"LOGISTIC REGRESSION RESULTS")
            print(f"{'='*60}")
            print(f"✅ Accuracy:  {accuracy:.3f}")
            print(f"✅ Precision: {precision:.3f}")
            print(f"✅ Recall:    {recall:.3f}")
            print(f"✅ F1-Score:  {f1:.3f}")
            print(f"✅ ROC-AUC:   {auc:.3f}")
            print(f"✅ Threshold: {self.threshold:.2f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  True Negatives: {cm[0,0]}  |  False Positives: {cm[0,1]}")
            print(f"  False Negatives: {cm[1,0]} |  True Positives: {cm[1,1]}")
            
            # Feature coefficients
            if hasattr(self.model, 'coef_'):
                coefficients = dict(zip(self.feature_columns, self.model.coef_[0]))
                print(f"\nTop 5 Features (Absolute coefficient):")
                top_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                for feat, coef in top_features:
                    direction = "↑ CHURN" if coef > 0 else "↓ RETAIN"
                    print(f"   • {feat}: {abs(coef):.3f} ({direction})")
            
            self.is_trained = True
            
            # Save model
            model_data = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'threshold': self.threshold
            }
            with open('logistic_regression_final.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"\n✅ Model saved to logistic_regression_final.pkl")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_model(self):
        try:
            if os.path.exists('logistic_regression_final.pkl'):
                with open('logistic_regression_final.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.label_encoders = model_data['label_encoders']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                self.training_metrics = model_data['training_metrics']
                self.threshold = model_data.get('threshold', 0.35)
                self.is_trained = True
                print(f"✅ Loaded Logistic Regression model")
                print(f"   Recall: {self.training_metrics['recall']:.3f}")
                return True
        except Exception as e:
            print(f"Load error: {e}")
        return False
    
    def predict(self, customer_data: Dict) -> Dict:
        if not self.is_trained:
            if not self.load_model():
                return self._fallback_predict()
        
        try:
            input_df = pd.DataFrame([customer_data])
            df_processed = self.prepare_features(input_df, fit_encoders=False)
            
            for col in self.feature_columns:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            
            X_input = df_processed[self.feature_columns].values
            X_input = np.nan_to_num(X_input)
            X_input_scaled = self.scaler.transform(X_input)
            
            churn_prob = self.model.predict_proba(X_input_scaled)[0][1]
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_predict()
        
        # Risk levels
        if churn_prob >= 0.7:
            risk = "High"
            priority = "Immediate Action"
        elif churn_prob >= 0.5:
            risk = "Medium"
            priority = "Monitor Closely"
        else:
            risk = "Low"
            priority = "Normal"
        
        return {
            "churn_probability": round(churn_prob, 3),
            "risk_level": risk,
            "action_priority": priority,
            "confidence": round(self.training_metrics.get('accuracy', 0.7), 2),
            "algorithm": "Logistic Regression",
            "key_factors": self._get_factors(customer_data),
            "recommendations": self._get_recommendations(risk),
            "interpretation": f"Model recall: {self.training_metrics['recall']*100:.0f}%"
        }
    
    def _get_factors(self, data):
        factors = []
        if data.get('nps_score', 5) <= 5:
            factors.append(f"NPS: {data.get('nps_score', 5)}/10")
        if data.get('content_satisfaction', 3) <= 3:
            factors.append(f"Satisfaction: {data.get('content_satisfaction', 3)}/5")
        if data.get('usage_frequency', '') == 'Rarely':
            factors.append("Rare usage")
        return factors if factors else ["No immediate risks"]
    
    def _get_recommendations(self, risk):
        if risk == "High":
            return ["🚨 Immediate retention call", "Offer discount", "Personal outreach"]
        elif risk == "Medium":
            return ["Send survey", "Loyalty rewards", "Engagement campaign"]
        else:
            return ["Regular check-ins", "Content recommendations", "Monitor activity"]
    
    def _fallback_predict(self):
        return {
            "churn_probability": 0.5,
            "risk_level": "Medium",
            "action_priority": "Monitor",
            "confidence": 0.5,
            "algorithm": "Fallback",
            "key_factors": ["Model not trained"],
            "recommendations": ["Upload CSV via /api/train"],
            "interpretation": "Train model first"
        }

# Initialize predictor
predictor = ChurnPredictor()

print("\n" + "="*60)
print("🚀 LOGISTIC REGRESSION CHURN API")
print("="*60)

if os.path.exists('logistic_regression_final.pkl'):
    predictor.load_model()
elif os.path.exists('OTT Platform Survey.csv'):
    print("\n📊 Training Logistic Regression...")
    predictor.train_model('OTT Platform Survey.csv')
else:
    print("\n⚠️ No data found. Upload CSV via POST /api/train")

print("="*60 + "\n")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Logistic Regression Churn API",
        "status": "running",
        "model_trained": predictor.is_trained,
        "recall": predictor.training_metrics.get('recall', 0) if predictor.is_trained else None
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_trained": predictor.is_trained}

@app.post("/api/train")
async def train_model(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    success = predictor.train_model(file_location)
    os.remove(file_location)
    return {"success": success, "metrics": predictor.training_metrics if success else None}

@app.get("/api/model/info")
async def model_info():
    return {"trained": predictor.is_trained, "metrics": predictor.training_metrics if predictor.is_trained else None}

@app.post("/api/data/customer")
async def submit_customer(customer: CustomerData):
    customer_id = str(uuid.uuid4())
    prediction = predictor.predict(customer.dict())
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO customers (id, age_group, primary_platform, subscription_duration,
        subscription_plan, monthly_spend, usage_frequency, nps_score,
        content_satisfaction, value_perception, churn_likelihood,
        switching_behavior, account_sharing, competitor_awareness,
        bundling_preference, payment_method, created_at,
        churn_probability, risk_level, prediction_confidence)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (customer_id, customer.age_group, customer.primary_platform,
          customer.subscription_duration, customer.subscription_plan,
          customer.monthly_spend, customer.usage_frequency, customer.nps_score,
          customer.content_satisfaction, customer.value_perception,
          customer.churn_likelihood, customer.switching_behavior,
          customer.account_sharing, customer.competitor_awareness,
          customer.bundling_preference, customer.payment_method,
          datetime.now().isoformat(), prediction['churn_probability'],
          prediction['risk_level'], prediction['confidence']))
    
    conn.commit()
    conn.close()
    return {"success": True, "customer_id": customer_id, "prediction": prediction}

@app.get("/api/data/customers")
async def get_customers(limit: int = 100):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, age_group, primary_platform, nps_score, churn_probability, risk_level, created_at FROM customers ORDER BY created_at DESC LIMIT ?', (limit,))
    customers = [dict(zip(['id', 'age_group', 'primary_platform', 'nps_score', 'churn_probability', 'risk_level', 'created_at'], row)) for row in cursor.fetchall()]
    conn.close()
    return {"success": True, "customers": customers}

@app.get("/api/stats")
async def get_stats():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(churn_probability) FROM customers")
    avg_churn = cursor.fetchone()[0] or 0
    cursor.execute("SELECT risk_level, COUNT(*) FROM customers GROUP BY risk_level")
    risk_dist = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return {"success": True, "stats": {"total": total, "avg_churn": round(avg_churn*100,1), "risk_distribution": risk_dist}}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Server: http://127.0.0.1:5001")
    print("📚 Docs: http://127.0.0.1:5001/docs")
    uvicorn.run(app, host="127.0.0.1", port=5002)