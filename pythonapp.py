print("Hello World!")

import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv("/exampleRepository/dataset01.csv")

data.head()

entries = data['y'].count()
print(f"Entries in column 'y': {entries}")

mean_y = data['y'].mean()
print(f"Mean of column 'y': {mean_y}")

sd = data['y'].std()
print(f"Standard deviation of 'y': {sd}")

var = data['y'].var()
print(f"Variance of 'y': {var}")

min_y = data['y'].min()
max_y = data['y'].max()
print(f"The values of 'y' range from {min_y} to {max_y}")

x = data['x'] #predictor variable
y = data['y'] #dependent variable

x = sm.add_constant(x)

model = sm.OLS(y,x).fit()

print(model.summary())
