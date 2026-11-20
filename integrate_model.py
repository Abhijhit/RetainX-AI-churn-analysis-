# integrate_model.py - Connect ML model to your web app
import joblib
import pandas as pd
import numpy as np

class ChurnPredictor:
    def __init__(self):
        """Load the trained model"""
        self.model = joblib.load('C:\\ml_project\\churn_model.pkl')
        self.feature_names = joblib.load('C:\\ml_project\\feature_names.pkl')
        print("✅ ML Model loaded successfully!")
    
    def predict(self, customer_data):
        """
        Predict churn for a single customer
        customer_data: dictionary from your web form
        """
        # Create feature array with all zeros
        features = {col: 0 for col in self.feature_names}
        
        # Map form data to model features
        features['On a scale of 0-10, how likely are you to recommend your primary platform to a friend?'] = \
            int(customer_data.get('nps_score', 5))
        
        features['How satisfied are you with the CONTENT on your primary platform?'] = \
            int(customer_data.get('content_satisfaction', 3))
        
        # Age mapping
        age_map = {'Under 18': 1, '18-35': 2, '35+': 3}
        features['age_encoded'] = age_map.get(customer_data.get('age_group', '18-35'), 2)
        
        # Duration mapping
        duration_map = {
            'Less than 1 month': 1, '1-3 months': 2, '3-6 months': 3,
            '6-12 months': 4, '1-2 years': 5, 'More than 2 years': 6
        }
        features['duration_encoded'] = duration_map.get(customer_data.get('subscription_duration', '1-3 months'), 2)
        
        # Usage mapping
        usage_map = {
            'Rarely (once in 2-3 months)': 1,
            'Often': 2,
            'Daily': 3
        }
        features['usage_encoded'] = usage_map.get(customer_data.get('usage_frequency', 'Often'), 2)
        
        # Value mapping
        value_map = {'Poor value': 1, 'Neutral': 2, 'Excellent value': 3}
        features['value_encoded'] = value_map.get(customer_data.get('value_perception', 'Neutral'), 2)
        
        # Platform mapping (if provided)
        if 'primary_platform' in customer_data:
            platform = customer_data['primary_platform']
            platform_col = f'platform_{platform[:15].replace(" ", "_").replace("+", "plus")}'
            if platform_col in self.feature_names:
                features[platform_col] = 1
        
        # Convert to DataFrame
        df = pd.DataFrame([features])[self.feature_names]
        
        # Predict
        probability = self.model.predict_proba(df)[0][1]
        prediction = self.model.predict(df)[0]
        
        return {
            'churn_probability': float(probability),
            'risk_level': 'High' if probability > 0.6 else 'Medium' if probability > 0.3 else 'Low',
            'will_churn': bool(prediction),
            'confidence': float(max(self.model.predict_proba(df)[0]))
        }

# Test the integration
if __name__ == "__main__":
    # Simulate form data from your web app
    test_customer = {
        'nps_score': 4,
        'content_satisfaction': 2,
        'age_group': '18-35',
        'subscription_duration': '1-3 months',
        'usage_frequency': 'Rarely (once in 2-3 months)',
        'value_perception': 'Poor value',
        'primary_platform': 'Netflix'
    }
    
    predictor = ChurnPredictor()
    result = predictor.predict(test_customer)
    
    print("\n🔮 PREDICTION RESULT:")
    print(f"   Churn Probability: {result['churn_probability']:.1%}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Will Churn: {result['will_churn']}")
    print(f"   Confidence: {result['confidence']:.1%}")