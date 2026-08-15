"""
representation_and_sequence_ablation.py

Two research questions, sharing one harness, both built from
Data/ATD_sequence.csv only (so every representation is computed from the
SAME 4,000 underlying samples -- a properly paired ablation, rather than
mixing in Data/ATD.csv's MEAN/RATIO columns, which come from a different,
larger sample set and can't be paired against E1-E50 truncations).

  Q1 "Does temporal information actually matter, or is it the
     architecture?" -- at each sequence length {5,10,20,30,40,50}, train
     BOTH a non-sequential MLP (flattened input) and a sequential LSTM on
     the identical truncated data. If LSTM > MLP at every length, that's
     architecture. If they track each other, the length itself is doing
     the work, not the recurrence.

  Q2 "How much information is enough?" (detection latency) -- LSTM
     accuracy/F1/AUC and inference cost as a function of sequence length.
     Answers: how many observations before the detector can decide?

  Q3 Representation richness independent of architecture -- the MLP is
     additionally run on two aggregate representations: MEAN (mean of
     all 50 raw values) and MEAN+RATIO_proxy (mean, max/mean). NOTE:
     Data/ATD.csv's RATIO column formula is not available in this repo
     (no matlab/ source), so RATIO here is an explicitly-labelled proxy
     (max/mean, a standard peak-to-average measure), computed from the
     same ATD_sequence.csv rows -- not a reproduction of ATD.csv's RATIO
     column, which comes from different, unpaired samples. This keeps
     the whole representation ladder on one consistent sample set.

Model capacity is held FIXED and deliberately compact (hidden=(32,16))
across every representation level for the MLP, and hidden_size=32 for
the LSTM (the Phase 1 winner) -- so representation is the only thing
that varies, not model capacity.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
OUT = "experiments/phase2_depth_robustness"
os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

dataset = pd.read_csv("Data/ATD_sequence.csv")
E = dataset.drop("LABEL", axis=1).values.astype(np.float32)  # (4000, 50)
y = dataset["LABEL"].values.astype(np.float32)

SEQ_LENGTHS = [5, 10, 20, 30, 40, 50]


def make_representation(n):
    """First n raw timesteps."""
    return E[:, :n]


def make_mean():
    return E.mean(axis=1, keepdims=True)


def make_mean_ratio_proxy():
    mean = E.mean(axis=1, keepdims=True)
    ratio_proxy = E.max(axis=1, keepdims=True) / (mean + 1e-8)
    return np.hstack([mean, ratio_proxy])


class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden=(32, 16)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                             dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(self.dropout(hidden[-1]))


def train_eval_mlp(X, y, epochs=30, seed=SEED):
    torch.manual_seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model = SmallMLP(X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1)),
                         batch_size=32, shuffle=True)
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    X_test_t = torch.tensor(X_test)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).numpy().flatten()
    latency = (time.perf_counter() - t0) / len(X_test) * 1000

    preds = (probs > 0.5).astype(float)
    return {
        "accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs), "latency_ms_per_sample": latency,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def train_eval_lstm(X, y, epochs=40, seed=SEED):
    torch.manual_seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model = LSTMClassifier(hidden_size=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train).unsqueeze(-1), torch.tensor(y_train).view(-1, 1)),
        batch_size=32, shuffle=True
    )
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    X_test_t = torch.tensor(X_test).unsqueeze(-1)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).numpy().flatten()
    latency = (time.perf_counter() - t0) / len(X_test) * 1000

    preds = (probs > 0.5).astype(float)
    return {
        "accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs), "latency_ms_per_sample": latency,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":

    rows = []

    print("=== Q3: representation ladder (MLP only, aggregate reps) ===")
    for name, X in [("MEAN", make_mean()), ("MEAN+RATIO_proxy", make_mean_ratio_proxy())]:
        m = train_eval_mlp(X, y)
        m.update({"representation": name, "n_features": X.shape[1], "model": "mlp"})
        rows.append(m)
        print(f"  {name} (d={X.shape[1]}): acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    print("\n=== Q1+Q2: sequence length ladder, MLP vs LSTM ===")
    for n in SEQ_LENGTHS:
        X_n = make_representation(n)

        m_mlp = train_eval_mlp(X_n, y)
        m_mlp.update({"representation": f"E1-E{n}", "n_features": n, "model": "mlp"})
        rows.append(m_mlp)

        m_lstm = train_eval_lstm(X_n, y)
        m_lstm.update({"representation": f"E1-E{n}", "n_features": n, "model": "lstm"})
        rows.append(m_lstm)

        print(f"  E1-E{n}: MLP acc={m_mlp['accuracy']:.4f} f1={m_mlp['f1']:.4f} auc={m_mlp['auc']:.4f} | "
              f"LSTM acc={m_lstm['accuracy']:.4f} f1={m_lstm['f1']:.4f} auc={m_lstm['auc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/results/representation_sequence_ablation.csv", index=False)

    # figure: accuracy vs sequence length, MLP vs LSTM
    seq_df = df[df["representation"].str.startswith("E1-E")].copy()
    seq_df["length"] = seq_df["representation"].str.replace("E1-E", "").astype(int)
    seq_df = seq_df.sort_values("length")

    plt.figure(figsize=(9, 6))
    for model_tag, marker in [("mlp", "o"), ("lstm", "s")]:
        sub = seq_df[seq_df["model"] == model_tag]
        plt.plot(sub["length"], sub["accuracy"], marker=marker, label=f"{model_tag.upper()} accuracy")
    plt.xlabel("Sequence length (timesteps retained)")
    plt.ylabel("Accuracy")
    plt.title("Detection accuracy vs. sequence length: architecture vs. information")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/sequence_length_architecture_comparison.png", dpi=300)
    plt.close()

    # figure: full representation ladder (MLP only) including aggregate reps
    ladder_order = ["MEAN", "MEAN+RATIO_proxy"] + [f"E1-E{n}" for n in SEQ_LENGTHS]
    ladder_df = df[(df["model"] == "mlp") & (df["representation"].isin(ladder_order))].copy()
    ladder_df["representation"] = pd.Categorical(ladder_df["representation"], categories=ladder_order, ordered=True)
    ladder_df = ladder_df.sort_values("representation")

    plt.figure(figsize=(10, 6))
    plt.plot(ladder_df["representation"].astype(str), ladder_df["accuracy"], marker="o", color="darkorange")
    plt.xticks(rotation=30)
    plt.ylabel("MLP accuracy")
    plt.title("Representation richness ladder (fixed compact MLP, same underlying samples)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/representation_ladder.png", dpi=300)
    plt.close()

    print(f"\nSaved {OUT}/results/representation_sequence_ablation.csv")
    print("Figures: sequence_length_architecture_comparison.png, representation_ladder.png")
