import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle 

# Step 1: Scrape data from the URL
url = "https://github.com/MarcusGrum/AIBAS/blob/main/README.md"  # Replace this with the actual URL
response = requests.get(url)

# Step 2: Parse the page content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Step 3: Find and extract the table
table = soup.find('table')
if table:
    df = pd.read_html(str(table))[0]  # Convert the table to a pandas DataFrame
    print("Table scraped successfully!")
else:
    print("No table found")

# Step 4: Clean the Data (handling missing values and stripping whitespace)
df = df.dropna()  # Drop rows with missing values
df['x'] = pd.to_numeric(df['x'], errors='coerce')  # Convert non-numeric values in 'x' to NaN
df['y'] = pd.to_numeric(df['y'], errors='coerce')  # Convert non-numeric values in 'y' to NaN

# Drop rows with NaN values after conversion
df = df.dropna()

# Step 5: Normalize the Data
scaler = StandardScaler()
df['normalized_x'] = scaler.fit_transform(df[['x']])  # Normalize the 'x' column
df['normalized_y'] = scaler.fit_transform(df[['y']])  # Normalize the 'y' column

# Step 6: Handle Outliers using Z-score
df['z_score_x'] = zscore(df['x'])
df['z_score_y'] = zscore(df['y'])

# Remove outliers (rows where z-score > 3)
df = df[(df['z_score_x'] < 3) & (df['z_score_y'] < 3)]

# Step 7: Rebuild the OLS Model with Cleaned Data
X = df[['x', 'normalized_x', 'normalized_y']]  # Independent variables
y = df['y']  # Dependent variable

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# Step 8: Visualize Results (scatter plot and regression line)
plt.scatter(df['x'], df['y'], color='blue')  # Scatter plot
plt.plot(df['x'], model.fittedvalues, color='red')  # OLS regression line
plt.xlabel('X')
plt.ylabel('Y')
plt.title('OLS Model: Scatter Plot with Regression Line')
plt.show()

# Step 9: Save the Cleaned Data and the Model
df.to_csv('cleaned_data.csv', index=False)  # Save cleaned data
with open('ols_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)  # Save the OLS model

print("Data cleaned and model built successfully!")
