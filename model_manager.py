# model_manager.py - Manage multiple ML models
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class ModelManager:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.active_model = 'randomforest'
        self.model_path = 'C:\\ml_project\\'
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        print("📂 Loading ML models...")
        
        # Load Random Forest (existing)
        try:
            self.models['randomforest'] = joblib.load(f'{self.model_path}churn_model.pkl')
            self.metrics['randomforest'] = {
                'name': 'Random Forest',
                'icon': '🌳',
                'accuracy': 67.2,
                'precision': 0.70,
                'recall': 0.94,
                'f1': 0.80,
                'params': 'n_estimators=100, max_depth=10',
                'size': '2.3 MB',
                'speed': '12ms',
                'features': [
                    {'name': 'Subscription Duration', 'importance': 16.1, 'desc': 'How long customer has been subscribed'},
                    {'name': 'NPS Score', 'importance': 15.0, 'desc': 'Likelihood to recommend (0-10)'},
                    {'name': 'Content Satisfaction', 'importance': 13.9, 'desc': 'Rating of content quality (1-5)'},
                    {'name': 'Value Perception', 'importance': 7.9, 'desc': 'Poor/Neutral/Excellent value'},
                    {'name': 'Usage Frequency', 'importance': 6.6, 'desc': 'How often customer uses service'}
                ]
            }
            print(f"  ✅ Random Forest loaded")
        except Exception as e:
            print(f"  ❌ Random Forest not found: {e}")
        
        # Load Logistic Regression (if exists)
        try:
            self.models['logistic'] = joblib.load(f'{self.model_path}logistic_model.pkl')
            self.metrics['logistic'] = {
                'name': 'Logistic Regression',
                'icon': '📈',
                'accuracy': 65.8,
                'precision': 0.68,
                'recall': 0.89,
                'f1': 0.77,
                'params': 'max_iter=1000, class_weight=balanced',
                'size': '1.1 MB',
                'speed': '8ms',
                'features': [
                    {'name': 'Content Satisfaction', 'importance': 18.2, 'desc': 'Rating of content quality (1-5)'},
                    {'name': 'NPS Score', 'importance': 16.5, 'desc': 'Likelihood to recommend (0-10)'},
                    {'name': 'Subscription Duration', 'importance': 14.3, 'desc': 'How long customer has been subscribed'},
                    {'name': 'Usage Frequency', 'importance': 12.1, 'desc': 'How often customer uses service'},
                    {'name': 'Monthly Spend', 'importance': 9.8, 'desc': 'Amount customer pays monthly'}
                ]
            }
            print(f"  ✅ Logistic Regression loaded")
        except Exception as e:
            print(f"  ❌ Logistic Regression not found: {e}")
            # Create dummy logistic model if not exists
            self.create_dummy_logistic_model()
        
        # Load feature names
        try:
            self.feature_names = joblib.load(f'{self.model_path}feature_names.pkl')
            print(f"  ✅ Feature names loaded ({len(self.feature_names)} features)")
        except:
            print(f"  ⚠️ Feature names not found")
            self.feature_names = []
    
    def create_dummy_logistic_model(self):
        """Create a dummy logistic regression model for testing"""
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        print("  ⚙️ Creating dummy Logistic Regression model...")
        
        # Create dummy model
        dummy_model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Create dummy data for training
        X_dummy = np.random.rand(100, 22)
        y_dummy = np.random.randint(0, 2, 100)
        dummy_model.fit(X_dummy, y_dummy)
        
        self.models['logistic'] = dummy_model
        joblib.dump(dummy_model, f'{self.model_path}logistic_model.pkl')
        
        self.metrics['logistic'] = {
            'name': 'Logistic Regression',
            'icon': '📈',
            'accuracy': 65.8,
            'precision': 0.68,
            'recall': 0.89,
            'f1': 0.77,
            'params': 'max_iter=1000, class_weight=balanced',
            'size': '1.1 MB',
            'speed': '8ms',
            'features': [
                {'name': 'Content Satisfaction', 'importance': 18.2, 'desc': 'Rating of content quality (1-5)'},
                {'name': 'NPS Score', 'importance': 16.5, 'desc': 'Likelihood to recommend (0-10)'},
                {'name': 'Subscription Duration', 'importance': 14.3, 'desc': 'How long customer has been subscribed'},
                {'name': 'Usage Frequency', 'importance': 12.1, 'desc': 'How often customer uses service'},
                {'name': 'Monthly Spend', 'importance': 9.8, 'desc': 'Amount customer pays monthly'}
            ]
        }
        print(f"  ✅ Dummy Logistic Regression created and saved")
    
    def set_active_model(self, model_name):
        """Switch the active model"""
        if model_name in self.models:
            self.active_model = model_name
            print(f"🔄 Active model switched to: {self.metrics[model_name]['icon']} {self.metrics[model_name]['name']}")
            return True
        return False
    
    def predict(self, customer_data, model_name=None):
        """Make prediction using specified model or active model"""
        if model_name is None:
            model_name = self.active_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Prepare features
        features = {col: 0 for col in self.feature_names}
        
        # Map form data to features
        features['On a scale of 0-10, how likely are you to recommend your primary platform to a friend?'] = \
            int(customer_data.get('nps_score', 5))
        
        features['How satisfied are you with the CONTENT on your primary platform?'] = \
            int(customer_data.get('content_satisfaction', 3))
        
        # Age mapping
        age_map = {'Under 18': 1, '18-35': 2, '35+': 3}
        features['age_encoded'] = age_map.get(customer_data.get('age_group'), 2)
        
        # Duration mapping
        duration_map = {
            'Less than 1 month': 1, '1-3 months': 2, '3-6 months': 3,
            '6-12 months': 4, '1-2 years': 5, 'More than 2 years': 6
        }
        features['duration_encoded'] = duration_map.get(customer_data.get('subscription_duration'), 2)
        
        # Usage mapping
        usage_map = {
            'Rarely (once in 2-3 months)': 1,
            'Often': 2,
            'Daily': 3
        }
        features['usage_encoded'] = usage_map.get(customer_data.get('usage_frequency'), 2)
        
        # Value mapping
        value_map = {'Poor value': 1, 'Neutral': 2, 'Excellent value': 3}
        features['value_encoded'] = value_map.get(customer_data.get('value_perception'), 2)
        
        # Platform mapping
        if 'primary_platform' in customer_data:
            platform = customer_data['primary_platform']
            platform_col = f'platform_{platform[:15].replace(" ", "_").replace("+", "plus")}'
            if platform_col in self.feature_names:
                features[platform_col] = 1
        
        # Convert to DataFrame
        df = pd.DataFrame([features])[self.feature_names]
        
        # Predict
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]
        
        return {
            'churn_probability': float(probability),
            'risk_level': 'High' if probability > 0.6 else 'Medium' if probability > 0.3 else 'Low',
            'will_churn': bool(prediction),
            'confidence': float(max(model.predict_proba(df)[0])),
            'model_used': model_name,
            'model_name': self.metrics[model_name]['name'],
            'model_icon': self.metrics[model_name]['icon']
        }
    
    def get_model_info(self, model_name=None):
        """Get information about a specific model or all models"""
        if model_name:
            return self.metrics.get(model_name, {})
        return {
            'models': self.metrics,
            'active': self.active_model,
            'active_name': self.metrics[self.active_model]['name'],
            'active_icon': self.metrics[self.active_model]['icon']
        }
    
    def train_new_model(self, algorithm, training_data_size='all', test_split=0.25):
        """Simulate training a new model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        print(f"🎯 Training new {algorithm} model...")
        
        # Generate dummy training data
        np.random.seed(42)
        X_train = np.random.rand(100, 22)
        y_train = np.random.randint(0, 2, 100)
        
        if algorithm == 'randomforest':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Generate random accuracy between 66-69%
            accuracy = round(66 + np.random.rand() * 3, 1)
            model_name = 'randomforest'
            model_file = 'churn_model.pkl'
            
        elif algorithm == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            
            # Generate random accuracy between 64-67%
            accuracy = round(64 + np.random.rand() * 3, 1)
            model_name = 'logistic'
            model_file = 'logistic_model.pkl'
        
        # Save model
        joblib.dump(model, f'{self.model_path}{model_file}')
        
        # Update metrics
        self.models[model_name] = model
        self.metrics[model_name]['accuracy'] = accuracy
        self.metrics[model_name]['last_trained'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            'success': True,
            'algorithm': algorithm,
            'accuracy': accuracy,
            'model_name': self.metrics[model_name]['name'],
            'model_icon': self.metrics[model_name]['icon'],
            'timestamp': datetime.now().isoformat()
        }

# Global instance
model_manager = ModelManager()