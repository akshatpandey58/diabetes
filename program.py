#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the diabetes dataset (assuming it's available as a CSV file)
# Replace 'your_dataset.csv' with the actual filename.
df_diabetes = pd.read_csv(r'C:\Users\Acer\OneDrive\Documents\diabetes.csv')

# Identify dependent and independent features
X = df_diabetes.drop('Outcome', axis=1)
y = df_diabetes['Outcome']

# Display information about the dataset
print("Number of records and features:", df_diabetes.shape)
print("Feature names:", df_diabetes.columns.tolist())
print("Number of records in each category of the 'Outcome' feature:")
print(df_diabetes['Outcome'].value_counts())

# Data Preprocessing
# Check for missing values
print("Missing values in the dataset:")
print(df_diabetes.isnull().sum())

# Splitting the dataset into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Building
# Declare the Decision Tree Classifier model
model_name = DecisionTreeClassifier(random_state=42)

# Fit the model
model_name.fit(X_train, y_train)

# Visualization of Tree Model
plt.figure(figsize=(12, 8))
from sklearn.tree import plot_tree
plot_tree(model_name, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Prediction of the Test set
y_pred = model_name.predict(X_test)

# Evaluation of the model
# Display the Classification Report and Confusion Matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Prediction for a New Unseen Record
# You can create a new record with input features and use the trained model to predict
# For example, if you want to predict for a new record:
new_record =  np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Replace with your own input data
new_prediction = model_name.predict(new_record)
print("Predicted outcome for the new record:", new_prediction)


# In[ ]:




