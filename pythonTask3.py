import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import statsmodels.api as sm
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic  # Ensure this file is in the same directory

# Step 1: Load Dataset
data = pd.read_csv('/tmp/exampleRepository/dataset02.csv')

# Debug: Check the dataset structure
print("Initial Data:")
print(data.head())
print(data.describe())

# Step 2: Data Cleaning
# Convert 'x' and 'y' to numeric and drop rows with errors or NaN values
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data = data.dropna()  # Remove rows with NaN values

# Debug: Check data after cleaning
print("\nData after Cleaning:")
print(data.describe())

# Step 3: Handle Outliers
# Option 1: Z-score filtering
z_scores = np.abs(zscore(data[['x', 'y']]))
data = data[(z_scores < 3).all(axis=1)]

# Debug: Check data size after Z-score filtering
print(f"\nNumber of rows after outlier removal: {data.shape[0]}")

# Option 2 (Optional): IQR filtering
# Uncomment this if Z-score filtering is not producing the desired results
Q1 = data[['x', 'y']].quantile(0.25)
Q3 = data[['x', 'y']].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[['x', 'y']] < (Q1 - 1.5 * IQR)) | (data[['x', 'y']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 4: Data Normalization
# Normalize 'x' and 'y' columns to the range [0, 1]
data[['x', 'y']] = (data[['x', 'y']] - data[['x', 'y']].min()) / (data[['x', 'y']].max() - data[['x', 'y']].min())

# Debug: Check data after normalization
print("\nData after Normalization:")
print(data.head())

# Step 5: Split Data into Training and Testing Sets
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)

# Save training and testing datasets
train.to_csv('/tmp/exampleRepository/dataset02_training.csv', index=False)
test.to_csv('/tmp/exampleRepository/dataset02_testing.csv', index=False)

# Debug: Save intermediate datasets for verification
train.to_csv('/tmp/exampleRepository/debug_training.csv', index=False)
test.to_csv('/tmp/exampleRepository/debug_testing.csv', index=False)

# Step 6: Scatter Plot with OLS Model
plt.figure(figsize=(10, 6))
plt.scatter(train['x'], train['y'], color='orange', label='Training Data')
plt.scatter(test['x'], test['y'], color='blue', label='Testing Data')

# Fit OLS model on training data
X_train = sm.add_constant(train['x'])  # Add constant for intercept
ols_model = sm.OLS(train['y'], X_train).fit()
train_predictions = ols_model.predict(X_train)

# Plot OLS model line
plt.plot(train['x'], train_predictions, color='red', label='OLS Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Scatter Plot with OLS Model')
plt.savefig('/tmp/exampleRepository/UE_04_App2_ScatterVisualizationAndOlsModel.pdf')
plt.close()

# Step 7: Boxplot of Data Dimensions
plt.figure(figsize=(8, 5))
data.boxplot(column=['x', 'y'])  # Plot only numeric columns
plt.title('Boxplot of Data Dimensions')
plt.savefig('/tmp/exampleRepository/UE_04_App2_BoxPlot.pdf')
plt.close()

# Step 8: Diagnostic Plots
# Save the OLS model summary
with open('/tmp/exampleRepository/OLS_model_summary.txt', 'w') as f:
    f.write(ols_model.summary().as_text())

# Generate diagnostic plots using the LinearRegDiagnostic class
diagnostic = LinearRegDiagnostic(ols_model)
_, fig, _ = diagnostic(plot_context='seaborn-talk', high_leverage_threshold=True, cooks_threshold='baseR')
fig.savefig('/tmp/exampleRepository/UE_04_App2_DiagnosticPlots.pdf')
plt.close()

# Debug: Save intermediate diagnostic plots
diagnostic.residual_plot().figure.savefig('/tmp/exampleRepository/Residuals_vs_Fitted.pdf')
diagnostic.qq_plot().figure.savefig('/tmp/exampleRepository/QQ_Plot.pdf')
diagnostic.scale_location_plot().figure.savefig('/tmp/exampleRepository/Scale_Location.pdf')
diagnostic.leverage_plot().figure.savefig('/tmp/exampleRepository/Residuals_vs_Leverage.pdf')
