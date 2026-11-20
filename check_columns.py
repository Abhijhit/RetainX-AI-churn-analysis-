# check_columns.py
import pandas as pd

df = pd.read_csv(r'C:\churn_data.csv')
print("📋 EXACT COLUMN NAMES:")
for col in df.columns:
    print(f'  "{col}"')