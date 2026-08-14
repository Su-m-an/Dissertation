import time

import pandas as pd
import numpy as np
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

# Select GPU if available, otherwise use CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}\n")

# Load the sequence dataset

dataset = pd.read_csv("Data/ATD_sequence.csv")

X = dataset.drop("LABEL", axis=1).values.astype(np.float32)
y = dataset["LABEL"].values.astype(np.float32)

# Hold out a genuine test set (both classes) that the model never trains on

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# The autoencoder only ever trains on normal sequences from the training split.
# A slice of those normal sequences is further held back purely to calibrate
# the anomaly threshold, so the threshold is never chosen using test-set data.

X_train_normal = X_train_full[y_train_full == 0]

X_fit, X_threshold_calib = train_test_split(
    X_train_normal,
    test_size=0.20,
    random_state=SEED
)

# Standardise using statistics from the training split only

scaler = StandardScaler()

X_fit = scaler.fit_transform(X_fit).astype(np.float32)
X_threshold_calib = scaler.transform(X_threshold_calib).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train_tensor = torch.tensor(X_fit)

X_threshold_tensor = torch.tensor(X_threshold_calib)

X_test_tensor = torch.tensor(X_test)

train_loader = DataLoader(
    TensorDataset(X_train_tensor),
    batch_size=32,
    shuffle=True
)

class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(50,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,8)

        )

        self.decoder = nn.Sequential(

            nn.Linear(8,16),
            nn.ReLU(),

            nn.Linear(16,32),
            nn.ReLU(),

            nn.Linear(32,50)

        )

    def forward(self,x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

model = Autoencoder().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 40

loss_history = []

print("Training Autoencoder...\n")

start = time.time()

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch in train_loader:

        inputs = batch[0].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs,inputs)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    loss_history.append(epoch_loss)

    print(f"Epoch {epoch+1:2d}/{epochs}   Loss = {epoch_loss:.6f}")

training_time = time.time() - start

print("\nTraining completed.")

# Calibrate the anomaly threshold using only held-out *normal* training
# data (never the test set), then apply that fixed threshold to the test set.

model.eval()

with torch.no_grad():

    calib_reconstructed = model(X_threshold_tensor.to(device)).cpu().numpy()

    reconstructed = model(X_test_tensor.to(device)).cpu().numpy()

calib_errors = np.mean(
    (X_threshold_calib - calib_reconstructed)**2,
    axis=1
)

threshold = np.percentile(calib_errors, 95)

errors = np.mean(
    (X_test - reconstructed)**2,
    axis=1
)

predictions = (errors > threshold).astype(int)

accuracy = accuracy_score(y_test,predictions)
precision = precision_score(y_test,predictions)
recall = recall_score(y_test,predictions)
f1 = f1_score(y_test,predictions)
auc = roc_auc_score(y_test,errors)

print("\nAccuracy :",round(accuracy,4))
print("Precision:",round(precision,4))
print("Recall   :",round(recall,4))
print("F1 Score :",round(f1,4))
print("AUC Score:",round(auc,4))

print("\nTraining Time:",round(training_time,2),"seconds")

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test,predictions))

print("\nClassification Report\n")

print(classification_report(y_test,predictions))

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
    "results/autoencoder_metrics.csv",
    index=False
)

torch.save(
    model.state_dict(),
    "saved_models/autoencoder.pth"
)

plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("Reconstruction Loss")

plt.title("Autoencoder Training Loss")

plt.grid(True)

plt.savefig(
    "figures/autoencoder_training_loss.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(8,5))

plt.hist(
    errors[y_test==0],
    bins=40,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    errors[y_test==1],
    bins=40,
    alpha=0.6,
    label="Attack"
)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    label="Threshold"
)

plt.xlabel("Reconstruction Error")

plt.ylabel("Frequency")

plt.title("Reconstruction Error Distribution")

plt.legend()

plt.grid(True)

plt.savefig(
    "figures/autoencoder_reconstruction_error.png",
    dpi=300
)

plt.close()

print("\nTraining curves saved.")

print("Model saved.")

print("Evaluation completed.")