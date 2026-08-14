import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("Data/ATD.csv")

# Separate the input features from the target labels
X = data[["MEAN", "RATIO"]]
y = data["LABEL"]

# Split the dataset into training and testing sets
# 80% of the data is used for training and 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

# Standardise the features so both variables are on the same scale.
# The scaler is fitted only on the training data and then applied
# to both training and testing datasets.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature scaling completed successfully.")

print("\nFirst five scaled training samples:\n")
print(X_train[:5])