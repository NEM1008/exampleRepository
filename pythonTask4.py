import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import pandas as pd
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer
import pickle

# Step 1: Load Dataset
data = pd.read_csv('/tmp/exampleRepository/dataset03.csv')

# Step 2: Data Preparation
# Convert columns to numeric and drop NaN values
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data = data.dropna()

# Split into training and testing sets
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)

# Step 3: Prepare the PyBrain Dataset
# Create supervised datasets for training and testing
train_ds = SupervisedDataSet(1, 1)  # 1 input (x), 1 output (y)
test_ds = SupervisedDataSet(1, 1)

for _, row in train.iterrows():
    train_ds.addSample([row['x']], [row['y']])

for _, row in test.iterrows():
    test_ds.addSample([row['x']], [row['y']])

# Step 4: Build the Neural Network
net = buildNetwork(1, 5, 1, hiddenclass=SigmoidLayer, bias=True)

# Step 5: Train the Network
trainer = BackpropTrainer(net, train_ds, learningrate=0.01, momentum=0.9)
for epoch in range(1000):  # Train for 1000 epochs
    error = trainer.train()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# Step 6: Test the Network
predictions = []
actuals = []

for input_data, target in test_ds:
    prediction = net.activate(input_data)
    predictions.append(prediction[0])
    actuals.append(target[0])

# Step 7: Save the Trained Model
with open('/tmp/exampleRepository/UE_05_App3_ANN_Model.pkl', 'wb') as model_file:
    pickle.dump(net, model_file)

# Step 8: Compare Outputs
print("Actual vs Predicted:")
for actual, pred in zip(actuals, predictions):
    print(f"Actual: {actual:.3f}, Predicted: {pred:.3f}")
