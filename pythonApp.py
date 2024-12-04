import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('/tmp/exampleRepository/dataset01.csv')

# Analyze 'y' column
y = data['y']

print(f"Number of entries in 'y': {len(y)}")
print(f"Mean of 'y': {np.mean(y)}")
print(f"Standard deviation of 'y': {np.std(y)}")
print(f"Variance of 'y': {np.var(y)}")
print(f"Minimum of 'y': {np.min(y)}, Maximum of 'y': {np.max(y)}")

# Simple OLS model
import statsmodels.api as sm
X = sm.add_constant(data['x'])  # Add a constant term
model = sm.OLS(y, X).fit()

print(model.summary())

# Save OLS model
with open('/tmp/OLS_model', 'w') as f:
    f.write(model.summary().as_text())

