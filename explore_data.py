# explore_data.py
import pandas as pd

print("="*50)
print("📊 CUSTOMER CHURN DATA EXPLORATION")
print("="*50)

# Read the CSV file
df = pd.read_csv(r'C:\churn_data.csv')

print(f"\n✅ Loaded {len(df)} customer records")
print(f"✅ {len(df.columns)} columns")

print("\n📋 COLUMN NAMES:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n✅ First 5 rows of data:")
print(df.head())

print("\n" + "="*50)
print("✅ SUCCESS! Your ML environment is ready!")
print("="*50)