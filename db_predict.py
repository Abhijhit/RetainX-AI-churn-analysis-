# db_predict.py - Connect database to ML model
import sqlite3
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

print("="*50)
print("🔄 CONNECTING DATABASE TO ML MODEL")
print("="*50)

# ============================================
# STEP 1: LOAD MODEL AND FEATURE NAMES
# ============================================
print("\n📂 LOADING MODEL...")
model = joblib.load('churn_model.pkl')
feature_names = joblib.load('feature_names.pkl')
print(f"   ✅ Model loaded with {len(feature_names)} features")

# ============================================
# STEP 2: CONNECT TO DATABASE
# ============================================
print("\n🗄️ CONNECTING TO DATABASE...")
conn = sqlite3.connect('backend/database.sqlite')
cursor = conn.cursor()

# ============================================
# STEP 3: FETCH CUSTOMERS WITHOUT PREDICTIONS
# ============================================
print("\n📊 FETCHING CUSTOMERS FROM DATABASE...")
query = """
SELECT id, age_group, subscription_duration, usage_frequency, 
       value_perception, nps_score, content_satisfaction,
       subscription_plan, monthly_spend, account_sharing,
       competitor_awareness, payment_method
FROM customers 
WHERE predicted_churn_probability IS NULL
"""

df = pd.read_sql_query(query, conn)
print(f"   📊 Found {len(df)} customers without predictions")

if len(df) == 0:
    print("   ⚠️ No new customers to predict. All have predictions!")
    conn.close()
    exit()

# ============================================
# STEP 4: TRANSFORM DATA TO MATCH MODEL FEATURES
# ============================================
print("\n🔄 TRANSFORMING DATA FOR MODEL...")

# Mapping functions for categorical variables
def encode_age(age_group):
    mapping = {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55+': 5}
    return mapping.get(age_group, 3)

def encode_duration(duration):
    mapping = {'<1 month': 1, '1-3 months': 2, '3-6 months': 3, 
               '6-12 months': 4, '1-2 years': 5, '>2 years': 6}
    return mapping.get(duration, 3)

def encode_usage(usage):
    mapping = {'Daily': 4, 'Weekly': 3, 'Monthly': 2, 'Rarely': 1}
    return mapping.get(usage, 2)

def encode_value(value):
    mapping = {'Excellent': 5, 'Good': 4, 'Average': 3, 'Poor': 2, 'Very Poor': 1}
    return mapping.get(value, 3)

# Apply encoding
df['age_encoded'] = df['age_group'].apply(encode_age)
df['duration_encoded'] = df['subscription_duration'].apply(encode_duration)
df['usage_encoded'] = df['usage_frequency'].apply(encode_usage)
df['value_encoded'] = df['value_perception'].apply(encode_value)

print(f"   ✅ Encoded {len(df)} records")

# ============================================
# STEP 5: CREATE PLATFORM COLUMNS (Placeholder)
# ============================================
print("\n🏷️ CREATING PLATFORM FEATURES...")

# Initialize all platform columns with 0
platform_columns = [f for f in feature_names if 'platform' in f]
for col in platform_columns:
    df[col] = 0

# Try to map subscription_plan to platform (adjust based on your data)
platform_mapping = {
    'Netflix': 'platform_Netflix',
    'Amazon Prime': 'platform_Amazon_Prime_vi',
    'Disney+': 'platform_Disneyplus_hotstar',
    'Jio Cinema': 'platform_Jio_Cinema',
    'YouTube Premium': 'platform_YouTube_Premium',
    'ZEE5': 'platform_ZEE_5',
    'Sony LIV': 'platform_Sony_LIV',
}

for idx, row in df.iterrows():
    plan = row.get('subscription_plan', '')
    for plan_name, platform_col in platform_mapping.items():
        if plan_name.lower() in str(plan).lower():
            if platform_col in df.columns:
                df.at[idx, platform_col] = 1
            break

print(f"   ✅ Created {len(platform_columns)} platform features")

# ============================================
# STEP 6: SELECT FEATURES IN CORRECT ORDER
# ============================================
print("\n📋 SELECTING FEATURES FOR PREDICTION...")

# Create feature array in correct order
X = pd.DataFrame()
for feature in feature_names:
    if feature in df.columns:
        X[feature] = df[feature]
    elif feature == 'age_encoded':
        X[feature] = df['age_encoded']
    elif feature == 'duration_encoded':
        X[feature] = df['duration_encoded']
    elif feature == 'usage_encoded':
        X[feature] = df['usage_encoded']
    elif feature == 'value_encoded':
        X[feature] = df['value_encoded']
    else:
        X[feature] = 0  # Default for missing features

print(f"   ✅ Feature matrix shape: {X.shape}")

# ============================================
# STEP 7: RUN PREDICTIONS
# ============================================
print("\n🔮 RUNNING CHURN PREDICTIONS...")
probabilities = model.predict_proba(X)[:, 1]  # Probability of churn
predictions = (probabilities >= 0.5).astype(int)

# Categorize risk
risk_categories = []
for prob in probabilities:
    if prob < 0.3:
        risk_categories.append('Low')
    elif prob < 0.6:
        risk_categories.append('Medium')
    else:
        risk_categories.append('High')

print(f"   ✅ Predictions completed")
print(f"   📊 High risk customers: {sum(probabilities >= 0.6)}")
print(f"   📊 Medium risk customers: {sum((probabilities >= 0.3) & (probabilities < 0.6))}")
print(f"   📊 Low risk customers: {sum(probabilities < 0.3)}")

# ============================================
# STEP 8: UPDATE DATABASE WITH PREDICTIONS
# ============================================
print("\n💾 UPDATING DATABASE WITH PREDICTIONS...")

updated = 0
for idx, row in df.iterrows():
    customer_id = row['id']
    prob = float(probabilities[idx])
    risk = risk_categories[idx]
    
    cursor.execute("""
        UPDATE customers 
        SET predicted_churn_probability = ?,
            churn_risk_category = ?,
            updated_at = ?
        WHERE id = ?
    """, (prob, risk, datetime.now(), customer_id))
    updated += 1

conn.commit()
print(f"   ✅ Updated {updated} customers with predictions")

# ============================================
# STEP 9: SHOW SUMMARY
# ============================================
print("\n" + "="*50)
print("📊 PREDICTION SUMMARY")
print("="*50)
print(f"   Total customers processed: {updated}")
print(f"   Average churn probability: {probabilities.mean():.2%}")
print(f"   High risk customers: {sum(probabilities >= 0.6)}")
print(f"   Medium risk customers: {sum((probabilities >= 0.3) & (probabilities < 0.6))}")
print(f"   Low risk customers: {sum(probabilities < 0.3)}")

# Close connection
conn.close()
print("\n✅ Database update complete!")
print("="*50)