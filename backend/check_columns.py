import pandas as pd

df = pd.read_csv('OTT Platform Survey.csv')
print('Actual columns:\n')
for i, col in enumerate(df.columns):
    print(f'{i}: "{col}"')
    
print('\n\nFirst few rows of data:')
print(df.head(3))
