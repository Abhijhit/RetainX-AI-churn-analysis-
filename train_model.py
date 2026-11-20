# train_model.py - Train your first churn prediction model
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*50)
print("🤖 TRAINING CHURN PREDICTION MODEL")
print("="*50)

# ============================================
# STEP 1: LOAD PREPARED DATA
# ============================================
print("\n📂 LOADING PREPARED DATA...")
X, y = joblib.load('C:\\ml_project\\prepared_data.pkl')
print(f"   • Features: {X.shape[1]}")
print(f"   • Samples: {X.shape[0]}")

# ============================================
# STEP 2: SPLIT DATA (Train/Test)
# ============================================
print("\n✂️ SPLITTING DATA...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   • Training set: {len(X_train)} customers")
print(f"   • Testing set: {len(X_test)} customers")

# ============================================
# STEP 3: TRAIN RANDOM FOREST MODEL
# ============================================
print("\n🌳 TRAINING RANDOM FOREST CLASSIFIER...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("   ✅ Model trained successfully!")

# ============================================
# STEP 4: EVALUATE MODEL
# ============================================
print("\n📊 EVALUATING MODEL PERFORMANCE...")

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 ACCURACY: {accuracy:.2%}")
print(f"   • Correct predictions: {accuracy_score(y_test, y_pred, normalize=False)} out of {len(y_test)}")
print(f"   • Wrong predictions: {len(y_test) - accuracy_score(y_test, y_pred, normalize=False)}")

print("\n📋 DETAILED REPORT:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

# ============================================
# STEP 5: FEATURE IMPORTANCE
# ============================================
print("\n🔍 TOP 5 FACTORS INFLUENCING CHURN:")
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance.head(5).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.1%}")

# ============================================
# STEP 6: SAVE MODEL
# ============================================
print("\n💾 SAVING MODEL...")
joblib.dump(model, 'C:\\ml_project\\churn_model.pkl')
print("   ✅ Model saved as 'churn_model.pkl'")

# Save feature names for later use
joblib.dump(X.columns.tolist(), 'C:\\ml_project\\feature_names.pkl')
print("   ✅ Feature names saved")

print("\n" + "="*50)
print("✅ MODEL TRAINING COMPLETE!")
print("="*50)