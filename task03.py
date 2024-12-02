import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.graphics.api as smg
from scipy.stats import zscore

# Load the dataset
data = pd.read_csv('dataset02.csv')

# Inspect the first few rows
print(data.head())

# Convert non-numeric columns to NaN
data_numeric = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
clean_data = data_numeric.dropna()

# Calculate Z-scores for outlier detection
z_scores = np.abs(zscore(clean_data))
print('\n Z-scores: ')
print(z_scores)

# Define a threshold to filter out outliers
threshold = 1
data_no_outliers = clean_data[(z_scores < threshold).all(axis=1)]
print("\n Data without outliers: ")
print(data_no_outliers)

Q1 = data_no_outliers.quantile(0.25)
Q3 = data_no_outliers.quantile(0.75)
IQR = Q3 - Q1
data_no_outliers = data_no_outliers[~((data_no_outliers < (Q1 - 1.5 * IQR)) | (data_no_outliers > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalize or Scale the data
# Min-Max scaling (scaling values to a range [0, 1])
data_scaled = (data_no_outliers - data_no_outliers.min()) / (data_no_outliers.max() - data_no_outliers.min())

# Or Standardization (Z-score normalization)
# data_scaled = data_no_outliers.apply(zscore)

# Now, data_scaled is ready for further analysis or model fitting

training_data = data_scaled.sample(frac=0.8, random_state=42)
training_data.to_csv('/exampleRepository/dataset02_training.csv',index=False)

testing_data = data_scaled.drop(training_data.index)
testing_data.to_csv('/exampleRepository/dataset02_testing.csv',index=False)

plt.scatter(training_data['x'], training_data['y'], color='blue', label='Training Data')
plt.scatter(testing_data['x'], testing_data['y'], color='orange', label='Testing Data')
plt.plot(training_data['x'], sm.OLS(training_data['y'], sm.add_constant(training_data[['x']])).fit().predict(sm.add_constant(training_data[['x']])), color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x vs y')
plt.legend()
plt.savefig('UE_04_App2_ScatterVisualizationAndOLSModel.pdf')
plt.show()

# Plotting (Example for scatter plot)
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=data_scaled['x'], y=data_scaled['y'], color='orange', label='Data')
# plt.title('Scatter Plot of Scaled Data')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()


data_no_outliers.boxplot()
plt.title('Box Plot of All Dimensions')
plt.savefig('UE_04_App2_BoxPlot.pdf')
plt.show()



model = sm.OLS(training_data['y'], sm.add_constant(training_data[['x']])).fit()
fig = smg.plot_regress_exog(model, 'x')
fig.savefig('UE_04_App2_DiagnosticPlots.pdf')
plt.show()