"""
mlp_regularization_ablation.py

Follow-up to mlp_low_snr_seed_check.py, which diagnosed why MLP fails at
the lowest SNR points in the sweep: train AUC 0.73, test AUC 0.11 at
rho_E=0.1, confirmed reproducible across 5 seeds. The Phase 1-tuned
architecture (hidden=(128,64,32), no dropout, no weight decay) was
selected for the signal-rich baseline regime and has enough capacity to
memorize noise when the true signal is nearly absent.

This tests whether standard regularization actually fixes that, rather
than assuming it would. Five configurations, each retrained with 5 seeds
at rho_E=0.1 and rho_E=0.5:

  1. baseline       - the original failing configuration, for comparison
  2. dropout_0.3     - same architecture, dropout 0.3 between layers
  3. dropout_0.5     - same architecture, dropout 0.5 between layers
  4. smaller         - hidden=(32,16) instead of (128,64,32), no dropout
  5. weight_decay    - same architecture, Adam weight_decay=1e-3
  6. smaller_dropout - hidden=(32,16) with dropout 0.3, combining both levers

The diagnostic that matters is the train/test AUC gap, not just test AUC
on its own, since a config could show a better test AUC by chance while
still overfitting badly underneath.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

P1 = "experiments/phase1_statistical_rigor/results"
OUT = "experiments/phase3_physical_layer"
SEEDS = [42, 43, 44, 45, 46]
RHO_E_POINTS = [0.1, 0.5]

os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

with open(f"{P1}/mlp_best_params.json") as f:
    mlp_params = json.load(f)

CONFIGS = {
    "baseline":        {"hidden": mlp_params["hidden"], "dropout": 0.0, "weight_decay": 0.0},
    "dropout_0.3":     {"hidden": mlp_params["hidden"], "dropout": 0.3, "weight_decay": 0.0},
    "dropout_0.5":     {"hidden": mlp_params["hidden"], "dropout": 0.5, "weight_decay": 0.0},
    "smaller":         {"hidden": [32, 16],              "dropout": 0.0, "weight_decay": 0.0},
    "weight_decay":    {"hidden": mlp_params["hidden"], "dropout": 0.0, "weight_decay": 1e-3},
    "smaller_dropout": {"hidden": [32, 16],              "dropout": 0.3, "weight_decay": 0.0},
}


class MLP(nn.Module):
    def __init__(self, hidden, dropout=0.0):
        super().__init__()
        layers, d = [], 50
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_eval(X_train, y_train, X_test, y_test, hidden, dropout, weight_decay, lr, epochs, seed):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1)),
                         batch_size=32, shuffle=True)
    model = MLP(hidden, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(torch.tensor(X_train))).numpy().flatten()
        test_prob = torch.sigmoid(model(torch.tensor(X_test))).numpy().flatten()

    return roc_auc_score(y_train, train_prob), roc_auc_score(y_test, test_prob)


rows = []
for rho_E in RHO_E_POINTS:
    dataset = pd.read_csv(f"{OUT}/raw_sweep/rho_E_{rho_E}/ATD_sequence.csv")
    X = dataset.drop("LABEL", axis=1).values.astype(np.float32)
    y = dataset["LABEL"].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    for config_name, cfg in CONFIGS.items():
        for seed in SEEDS:
            train_auc, test_auc = train_eval(
                X_train_s, y_train, X_test_s, y_test,
                hidden=cfg["hidden"], dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
                lr=mlp_params["lr"], epochs=30, seed=seed
            )
            gap = train_auc - test_auc
            rows.append({
                "rho_E": rho_E, "config": config_name, "seed": seed,
                "train_auc": train_auc, "test_auc": test_auc, "train_test_gap": gap,
            })
            print(f"  rho_E={rho_E} {config_name:16s} seed={seed}: "
                  f"train_auc={train_auc:.4f} test_auc={test_auc:.4f} gap={gap:.4f}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results/mlp_regularization_ablation.csv", index=False)

summary = df.groupby(["rho_E", "config"])[["train_auc", "test_auc", "train_test_gap"]].agg(["mean", "std"])
summary.to_csv(f"{OUT}/results/mlp_regularization_ablation_summary.csv")

print("\n=== Summary (mean across 5 seeds) ===")
for rho_E in RHO_E_POINTS:
    print(f"\nrho_E={rho_E}:")
    for config_name in CONFIGS:
        sub = df[(df["rho_E"] == rho_E) & (df["config"] == config_name)]
        print(f"  {config_name:16s} test_auc={sub['test_auc'].mean():.4f}+/-{sub['test_auc'].std():.4f}  "
              f"gap={sub['train_test_gap'].mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, rho_E in zip(axes, RHO_E_POINTS):
    sub = df[df["rho_E"] == rho_E]
    means = sub.groupby("config")["test_auc"].mean().reindex(CONFIGS.keys())
    stds = sub.groupby("config")["test_auc"].std().reindex(CONFIGS.keys())
    ax.bar(means.index, means.values, yerr=stds.values, capsize=4, color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUC=0.5)")
    ax.set_title(f"rho_E={rho_E}")
    ax.set_ylabel("Test AUC")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
plt.suptitle("MLP regularization ablation: does it fix the low-SNR overfitting failure?")
plt.tight_layout()
plt.savefig(f"{OUT}/figures/mlp_regularization_ablation.png", dpi=300)
plt.close()

print(f"\nSaved {OUT}/results/mlp_regularization_ablation.csv")
print(f"Saved {OUT}/results/mlp_regularization_ablation_summary.csv")
print(f"Saved {OUT}/figures/mlp_regularization_ablation.png")
