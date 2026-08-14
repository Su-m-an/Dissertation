"""
01_data_loading.py

Purpose:
Load the Artificial Training Dataset (ATD) generated in MATLAB
and perform a basic inspection before preprocessing.

Author: Suman R
"""

import pandas as pd

print("\nLoading the Artificial Training Dataset...\n")

# Read the dataset generated from MATLAB
data = pd.read_csv("Data/ATD.csv")

print("Dataset loaded successfully.\n")

# Display the size of the dataset
print(f"Dataset Shape: {data.shape}")

# Display the column names
print("\nColumn Names:")
print(data.columns.tolist())

# Check the data type of each column
print("\nData Types:")
print(data.dtypes)

# Display the first five rows
print("\nFirst Five Rows:")
print(data.head())

# Check if there are any missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check how many samples belong to each class
print("\nClass Distribution:")
print(data["LABEL"].value_counts())

# Display some basic statistics for the numerical features
print("\nSummary Statistics:")
print(data.describe())