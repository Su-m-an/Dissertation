"""
13_lstm.py

Superseded by 13_lstm_final.py, which adds a validation split and early
stopping. results/lstm_metrics.csv and figures/lstm_training_loss.png get
overwritten by whichever of the two scripts runs last. The current
results/ files were produced by 13_lstm_final.py. Kept for reference only;
prefer 13_lstm_final.py.
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {device}\n")

dataset = pd.read_csv("Data/ATD_sequence.csv")

X = dataset.drop("LABEL", axis=1).values.astype(np.float32)
y = dataset["LABEL"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# Standardise the sequence features before feeding them to the LSTM

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train = torch.tensor(X_train).unsqueeze(-1)
X_test = torch.tensor(X_test).unsqueeze(-1)

y_train = torch.tensor(y_train).view(-1,1)
y_test = torch.tensor(y_test).view(-1,1)

train_dataset = TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test,y_test)

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

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        output,(hidden,cell)=self.lstm(x)

        output = hidden[-1]

        output = self.fc(output)

        return output

model = LSTMClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 40

best_loss = np.inf

loss_history = []

print("Training LSTM...\n")

start = time.time()

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for inputs,labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss/len(train_loader)

    loss_history.append(epoch_loss)

    print(f"Epoch {epoch+1:2d}/{epochs} Loss={epoch_loss:.4f}")

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            "saved_models/lstm_best.pth"
        )

training_time = time.time()-start

print("\nLoading Best Model...\n")

model.load_state_dict(
    torch.load(
        "saved_models/lstm_best.pth",
        map_location=device
    )
)

model.eval()

predictions = []
probabilities = []
actual = []

with torch.no_grad():

    for inputs,labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        probs = torch.sigmoid(outputs)

        preds = (probs>0.5).float()

        predictions.extend(preds.cpu().numpy())

        probabilities.extend(probs.cpu().numpy())

        actual.extend(labels.numpy())

predictions = np.array(predictions).flatten()
probabilities = np.array(probabilities).flatten()
actual = np.array(actual).flatten()

accuracy = accuracy_score(actual,predictions)
precision = precision_score(actual,predictions)
recall = recall_score(actual,predictions)
f1 = f1_score(actual,predictions)
auc = roc_auc_score(actual,probabilities)

print("\nResults\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

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
        "AUC"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]

})

metrics.to_csv(
    "results/lstm_metrics.csv",
    index=False
)

plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Loss")

plt.grid(True)

plt.savefig(
    "figures/lstm_training_loss.png",
    dpi=300
)

plt.close()

print("\nTraining curve saved.")

print("LSTM model saved.")