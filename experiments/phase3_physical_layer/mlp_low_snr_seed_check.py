"""
mlp_low_snr_seed_check.py

Follow-up triggered by an anomaly in the main SNR sweep: MLP showed
below-chance AUC at rho_E=0.1 (0.11) and rho_E=0.5 (0.44), while TC-SVM/
RF/XGBoost/LSTM/Autoencoder all looked sensible (AUC ~0.5) at the same
points. Since each sweep point trains once with no retry, this could be
a genuine architecture-level instability at very weak signal, or just an
unlucky single run. Retrains MLP at rho_E=0.1 and 0.5 with 5 different
seeds each to tell those apart before reporting it as a finding.
"""

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import json

P1 = "experiments/phase1_statistical_rigor/results"
OUT = "experiments/phase3_physical_layer"
SEEDS = [42, 43, 44, 45, 46]

os.makedirs(f"{OUT}/results", exist_ok=True)

with open(f"{P1}/mlp_best_params.json") as f:
    mlp_params = json.load(f)


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers, d = [], 50
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_eval_mlp(X_train, y_train, X_test, y_test, hidden, lr, epochs, seed):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1)),
                         batch_size=32, shuffle=True)
    model = MLP(tuple(hidden))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(X_test))).numpy().flatten()
    pred = (prob > 0.5).astype(float)
    return accuracy_score(y_test, pred), roc_auc_score(y_test, prob)


rows = []
for rho_E in [0.1, 0.5]:
    dataset = pd.read_csv(f"{OUT}/raw_sweep/rho_E_{rho_E}/ATD_sequence.csv")
    X = dataset.drop("LABEL", axis=1).values.astype(np.float32)
    y = dataset["LABEL"].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    for seed in SEEDS:
        acc, auc = train_eval_mlp(X_train_s, y_train, X_test_s, y_test,
                                    mlp_params["hidden"], mlp_params["lr"], 30, seed)
        rows.append({"rho_E": rho_E, "seed": seed, "accuracy": acc, "auc": auc})
        print(f"  rho_E={rho_E} seed={seed}: acc={acc:.4f} auc={auc:.4f}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results/mlp_low_snr_seed_check.csv", index=False)

print("\n=== Summary ===")
for rho_E in [0.1, 0.5]:
    sub = df[df["rho_E"] == rho_E]
    print(f"rho_E={rho_E}: AUC mean={sub['auc'].mean():.4f} std={sub['auc'].std():.4f} "
          f"min={sub['auc'].min():.4f} max={sub['auc'].max():.4f}")

print(f"\nSaved {OUT}/results/mlp_low_snr_seed_check.csv")
