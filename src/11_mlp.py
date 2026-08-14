import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

print("\nLoading Sequence Dataset...\n")

dataset = pd.read_csv("Data/ATD_sequence.csv")

X = dataset.drop("LABEL", axis=1).values.astype(np.float32)
y = dataset["LABEL"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)

y_train = torch.tensor(y_train).view(-1,1)
y_test = torch.tensor(y_test).view(-1,1)

train_dataset = TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test,y_test)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

print("Building MLP...\n")

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(50,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)

        )

    def forward(self,x):

        return self.network(x)

model = MLP()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 30

loss_history = []

print("Training MLP...\n")

start_time = time.time()

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for inputs,labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss/len(train_loader)

    loss_history.append(epoch_loss)

    print(f"Epoch {epoch+1:2d}/{epochs}   Loss = {epoch_loss:.4f}")

training_time = time.time()-start_time

print("\nTraining completed.\n")

model.eval()

predictions = []

probabilities = []

actual = []

with torch.no_grad():

    for inputs,labels in test_loader:

        outputs = model(inputs)

        probs = torch.sigmoid(outputs)

        preds = (probs>0.5).float()

        predictions.extend(preds.numpy())

        probabilities.extend(probs.numpy())

        actual.extend(labels.numpy())

predictions = np.array(predictions).flatten()

probabilities = np.array(probabilities).flatten()

actual = np.array(actual).flatten()

accuracy = accuracy_score(actual,predictions)
precision = precision_score(actual,predictions)
recall = recall_score(actual,predictions)
f1 = f1_score(actual,predictions)
auc = roc_auc_score(actual,probabilities)

print("Accuracy :",round(accuracy,4))
print("Precision:",round(precision,4))
print("Recall   :",round(recall,4))
print("F1 Score :",round(f1,4))
print("AUC Score:",round(auc,4))

print("\nTraining Time:",round(training_time,2),"seconds")

print("\nConfusion Matrix\n")

print(confusion_matrix(actual,predictions))

print("\nClassification Report\n")

print(classification_report(actual,predictions))

metrics = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC",
        "Training Time (s)"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        auc,
        training_time
    ]

})

metrics.to_csv(
    "results/mlp_metrics.csv",
    index=False
)

print("\nMetrics saved successfully.")

plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("MLP Training Loss")

plt.grid(True)

plt.savefig("figures/mlp_training_loss.png",dpi=300)

plt.close()

print("\nTraining curve saved successfully.")