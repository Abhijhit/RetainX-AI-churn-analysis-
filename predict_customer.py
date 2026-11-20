# predict_customer.py - Predict churn for ONE customer
import pandas as pd
import joblib
import numpy as np

print("="*50)
print("🔮 TESTING MODEL ON SINGLE CUSTOMER")
print("="*50)

# Load model and feature names
model = joblib.load('C:\\ml_project\\churn_model.pkl')
feature_names = joblib.load('C:\\ml_project\\feature_names.pkl')

print(f"\n📋 Model uses {len(feature_names)} features:")
for i, feat in enumerate(feature_names[:10]):
    print(f"   {i+1}. {feat}")
if len(feature_names) > 10:
    print(f"   ... and {len(feature_names)-10} more")

# Create a sample customer with ALL features set to 0 first
sample_data = {col: 0 for col in feature_names}

# Now set values for the features we know
sample_data['On a scale of 0-10, how likely are you to recommend your primary platform to a friend?'] = 4
sample_data['How satisfied are you with the CONTENT on your primary platform?'] = 2
sample_data['age_encoded'] = 2
sample_data['duration_encoded'] = 2
sample_data['usage_encoded'] = 1
sample_data['value_encoded'] = 1

# Set platform columns that exist in your data
platform_mapping = {
    'platform_Amazon_Prime_vi': 1,
    'platform_Netflix': 1,
    'platform_Disneyplus_hotstar': 0,
    'platform_Jio_Cinema': 0,
}

for plat, val in platform_mapping.items():
    if plat in feature_names:
        sample_data[plat] = val

# Convert to DataFrame with correct column order
sample_df = pd.DataFrame([sample_data])[feature_names]

# Predict
pred = model.predict(sample_df)[0]
prob = model.predict_proba(sample_df)[0]

print("\n" + "="*50)
print("🎯 PREDICTION RESULT:")
print("="*50)
print(f"\n📊 Churn Probability: {prob[1]:.1%}")
print(f"🔮 Prediction: {'🚨 WILL CHURN' if pred == 1 else '✅ WILL STAY'}")
print(f"⚠️ Risk Level: {'HIGH' if prob[1] > 0.5 else 'LOW'}")

print("\n" + "="*50)
print("✅ MODEL READY FOR INTEGRATION!")
print("="*50)