import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

print("\nLoading the Sequence Dataset...\n")

dataset = pd.read_csv("Data/ATD_sequence.csv")

print("Dataset loaded successfully.\n")

print("Dataset Shape:")
print(dataset.shape)

print("\nColumn Names:")
print(dataset.columns.tolist())

print("\nFirst Five Rows:")
print(dataset.head())

print("\nMissing Values:")
print(dataset.isnull().sum())

print("\nClass Distribution:")
print(dataset["LABEL"].value_counts())

X = dataset.drop("LABEL", axis=1)

y = dataset["LABEL"]

X = X.values.astype(np.float32)

y = y.values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

X_train_tensor = torch.tensor(X_train)

X_test_tensor = torch.tensor(X_test)

y_train_tensor = torch.tensor(y_train)

y_test_tensor = torch.tensor(y_test)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("\nPyTorch DataLoaders created successfully.")

print("\nTraining Tensor Shape:")
print(X_train_tensor.shape)

print("\nTesting Tensor Shape:")
print(X_test_tensor.shape)

print("\nExample Sequence")

print(X_train_tensor[0])

print("\nCorresponding Label")

print(y_train_tensor[0])