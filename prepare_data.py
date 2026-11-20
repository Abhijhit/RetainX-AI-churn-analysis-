# prepare_data.py - Convert text answers to numbers for ML
import pandas as pd
import numpy as np

print("="*50)
print("🔄 PREPARING DATA FOR MACHINE LEARNING")
print("="*50)

# Load your data
df = pd.read_csv(r'C:\churn_data.csv')
print(f"\n✅ Loaded {len(df)} customer records")

# ============================================
# STEP 1: CREATE TARGET VARIABLE (What we predict)
# ============================================
print("\n🎯 CREATING TARGET VARIABLE...")

# EXACT column name with two spaces at the end
churn_col = 'How likely are you to cancel your primary OTT subscription in the next 3 months?  '

# Convert to binary classification: 1 = Will churn, 0 = Will stay
df['churn'] = df[churn_col].map({
    'Very unlikely': 0,
    'Not sure': 1,
    'Very likely': 1
})

print(f"   • Customers staying (0): {len(df[df['churn']==0])}")
print(f"   • Customers at risk (1): {len(df[df['churn']==1])}")

# ============================================
# STEP 2: ENCODE CATEGORICAL FEATURES
# ============================================
print("\n🏷️ ENCODING CATEGORICAL FEATURES...")

# Age group mapping (ordinal)
age_map = {
    'Under 18': 1,
    '18-35': 2,
    '35+': 3
}
df['age_encoded'] = df['Which age group do you belong to?'].map(age_map)

# Subscription duration mapping (ordinal)
duration_map = {
    'Less than 1 month': 1,
    '1-3 months': 2,
    '3-6 months': 3,
    '6-12 months': 4,
    '1-2 years': 5,
    'More than 2 years': 6
}
df['duration_encoded'] = df['How long have you been continuously subscribed to your primary platform?'].map(duration_map)

# Usage frequency mapping (ordinal)
usage_map = {
    'Rarely (once in 2-3 months)': 1,
    'Often': 2,
    'Daily': 3
}
df['usage_encoded'] = df['How often do you actually use your primary subscription?'].map(usage_map)

# Value perception mapping (ordinal)
value_map = {
    'Poor value': 1,
    'Neutral': 2,
    'Excellent value': 3
}
df['value_encoded'] = df['Do you feel you\'re getting good value for what you pay?'].map(value_map)

# ============================================
# STEP 3: HANDLE MULTI-SELECT PLATFORMS
# ============================================
print("\n📱 PROCESSING PLATFORM SELECTIONS...")

# Split the semicolon-separated platforms
platforms_col = 'Which OTT platform(s) are you currently subscribed to?'

# Get all unique platforms
all_platforms = set()
for val in df[platforms_col].dropna():
    if ';' in str(val):
        all_platforms.update([p.strip() for p in str(val).split(';')])
    else:
        all_platforms.add(str(val).strip())

# Create binary columns for each platform
for platform in all_platforms:
    col_name = f'platform_{platform[:15].replace(" ", "_").replace("+", "plus")}'
    df[col_name] = df[platforms_col].apply(
        lambda x: 1 if platform in str(x) else 0
    )
print(f"   • Created {len(all_platforms)} platform columns")

# ============================================
# STEP 4: SELECT FINAL FEATURES FOR ML
# ============================================
print("\n🎯 SELECTING ML FEATURES...")

feature_columns = [
    # Numerical features (already numbers)
    'On a scale of 0-10, how likely are you to recommend your primary platform to a friend?',
    'How satisfied are you with the CONTENT on your primary platform?',
    
    # Encoded categorical features
    'age_encoded',
    'duration_encoded', 
    'usage_encoded',
    'value_encoded',
]

# Add all platform columns
platform_cols = [col for col in df.columns if col.startswith('platform_')]
feature_columns.extend(platform_cols)

# Create feature matrix X and target y
X = df[feature_columns].copy()
y = df['churn'].copy()

print(f"\n✅ FINAL DATASET SHAPE:")
print(f"   • Features (X): {X.shape[1]} columns")
print(f"   • Samples: {X.shape[0]} customers")
print(f"   • Target (y): {y.name}")

# ============================================
# STEP 5: SAVE PREPARED DATA
# ============================================
print("\n💾 SAVING PREPARED DATA...")

# Save as CSV for inspection
X.to_csv('C:\\ml_project\\features.csv', index=False)
y.to_csv('C:\\ml_project\\target.csv', index=False)

# Save as pickle for ML
import joblib
joblib.dump((X, y), 'C:\\ml_project\\prepared_data.pkl')

print("   ✅ features.csv saved")
print("   ✅ target.csv saved") 
print("   ✅ prepared_data.pkl saved")

print("\n" + "="*50)
print("✅ DATA PREPARATION COMPLETE!")
print("="*50)
print("\n📊 FEATURES BEING USED:")
for i, col in enumerate(X.columns[:10], 1):
    print(f"  {i}. {col}")
if len(X.columns) > 10:
    print(f"  ... and {len(X.columns)-10} more features")