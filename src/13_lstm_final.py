import time
import copy

import numpy as np
import pandas as pd
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {device}\n")

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

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

X_train = torch.tensor(X_train).unsqueeze(-1)
X_val = torch.tensor(X_val).unsqueeze(-1)
X_test = torch.tensor(X_test).unsqueeze(-1)

y_train = torch.tensor(y_train).view(-1,1)
y_val = torch.tensor(y_val).view(-1,1)
y_test = torch.tensor(y_test).view(-1,1)

train_loader = DataLoader(
    TensorDataset(X_train,y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val,y_val),
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test,y_test),
    batch_size=32,
    shuffle=False
)

class LSTMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        _,(hidden,_) = self.lstm(x)

        x = hidden[-1]

        x = self.dropout(x)

        x = self.fc(x)

        return x

model = LSTMClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 50

patience = 5

counter = 0

best_loss = np.inf

best_model = None

train_losses = []

val_losses = []

print("Training LSTM...\n")

start = time.time()

for epoch in range(epochs):

    model.train()

    train_loss = 0

    for inputs,labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    train_losses.append(train_loss)

    model.eval()

    validation_loss = 0

    with torch.no_grad():

        for inputs,labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs,labels)

            validation_loss += loss.item()

    validation_loss /= len(val_loader)

    val_losses.append(validation_loss)

    print(
        f"Epoch {epoch+1:2d}/{epochs} | "
        f"Train {train_loss:.4f} | "
        f"Validation {validation_loss:.4f}"
    )

    if validation_loss < best_loss:

        best_loss = validation_loss

        counter = 0

        best_model = copy.deepcopy(model.state_dict())

    else:

        counter += 1

    if counter >= patience:

        print("\nEarly stopping triggered.\n")

        break

training_time = time.time() - start

model.load_state_dict(best_model)

torch.save(
    model.state_dict(),
    "saved_models/lstm_best.pth"
)

print("\nBest model restored.\n")

model.eval()

predictions = []
probabilities = []
actual = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        predictions.extend(preds.cpu().numpy())

        probabilities.extend(probs.cpu().numpy())

        actual.extend(labels.numpy())

predictions = np.array(predictions).flatten()
probabilities = np.array(probabilities).flatten()
actual = np.array(actual).flatten()

accuracy = accuracy_score(actual, predictions)
precision = precision_score(actual, predictions)
recall = recall_score(actual, predictions)
f1 = f1_score(actual, predictions)
auc = roc_auc_score(actual, probabilities)

print("\nFinal Results\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print(f"\nTraining Time : {training_time:.2f} seconds")

print("\nConfusion Matrix\n")

cm = confusion_matrix(actual, predictions)

print(cm)

print("\nClassification Report\n")

print(classification_report(actual, predictions))

metrics = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
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
    "results/lstm_metrics.csv",
    index=False
)

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Training Loss")

plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("LSTM Training and Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "figures/lstm_training_validation_loss.png",
    dpi=300
)

plt.close()

torch.save(
    model.state_dict(),
    "saved_models/lstm_final.pth"
)

print("\nMetrics saved to results/")
print("Training curve saved to figures/")
print("Model saved to saved_models/")