"""
class_imbalance_stress_test.py

Re-runs all six models at realistic class imbalance (90/10, 95/5, 99/1,
99.9/0.1 attack ratios) instead of the artificial 50/50 balance of
ATD.csv / ATD_sequence.csv, to see whether the model comparison holds up
when attacks are actually rare. Accuracy is NOT the headline metric here:
at 99/1 a model that always predicts "no attack" scores 99% while
catching nothing, so precision, recall, F1, PR-AUC, FPR and FNR lead instead.

Design: the non-attack pool is held at its full available size (92,000
for the classical/ATD.csv models, 2,000 for the neural/ATD_sequence.csv
models); the attack class is subsampled to hit the target ratio, then the
whole resampled pool gets a fresh stratified 80/20 split so BOTH train
and test reflect the target imbalance (evaluating "as deployed", not
training balanced and testing imbalanced). A ratio is skipped, with the
reason logged, when the resulting test-set attack count would fall below
15, which isn't enough to trust a rate estimate. This is why the neural models
only get 90/10 and 95/5: ATD_sequence.csv has only 2,000 attack rows
total to begin with.

Hyperparameters are fixed to the Phase 1 tuned values for every model
(experiments/phase1_statistical_rigor/results/*_best_params.json).
This experiment varies only the training/test data distribution, not
the architecture search, which would be its own (expensive) nested study.
"""

import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    accuracy_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
P1 = "experiments/phase1_statistical_rigor/results"
OUT = "experiments/phase2_depth_robustness"
MIN_TEST_POSITIVES = 15

RATIOS = [0.10, 0.05, 0.01, 0.001]  # attack fraction: 90/10, 95/5, 99/1, 99.9/0.1

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_best_params(tag):
    with open(f"{P1}/{tag}_best_params.json") as f:
        return json.load(f)


def rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    return fpr, fnr


def resample_to_ratio(X, y, ratio, seed=SEED):
    """Fix the non-attack pool at full size; subsample attack to hit ratio."""
    rng = np.random.RandomState(seed)
    non_attack_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    n_non_attack = len(non_attack_idx)
    n_attack_target = int(round(n_non_attack * ratio / (1 - ratio)))
    n_attack_target = min(n_attack_target, len(attack_idx))

    chosen_attack = rng.choice(attack_idx, size=n_attack_target, replace=False)
    all_idx = np.concatenate([non_attack_idx, chosen_attack])
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def eval_metrics(y_true, y_pred, y_prob):
    fpr, fnr = rates(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "fpr": fpr,
        "fnr": fnr,
    }


# --------------------------------------------------------- classical ---

def classical_model_factory(tag):
    params = load_best_params(tag)
    if tag == "tc_svm":
        return lambda: SVC(kernel="rbf", probability=True, random_state=SEED, **params)
    if tag == "random_forest":
        return lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    if tag == "xgboost":
        return lambda: XGBClassifier(random_state=SEED, eval_metric="logloss", **params)


def run_classical():
    data = pd.read_csv("Data/ATD.csv")
    X_all = data[["MEAN", "RATIO"]].values
    y_all = data["LABEL"].values

    rows = []
    for tag in ["tc_svm", "random_forest", "xgboost"]:
        factory = classical_model_factory(tag)
        for ratio in RATIOS:
            X_r, y_r = resample_to_ratio(X_all, y_all, ratio)
            X_train, X_test, y_train, y_test = train_test_split(
                X_r, y_r, test_size=0.20, random_state=SEED, stratify=y_r
            )
            n_test_pos = int(y_test.sum())
            if n_test_pos < MIN_TEST_POSITIVES:
                print(f"  SKIP {tag} @ ratio={ratio}: only {n_test_pos} test-set attacks (< {MIN_TEST_POSITIVES})")
                rows.append({"model": tag, "ratio": ratio, "skipped": True,
                             "reason": f"only {n_test_pos} test-set positives"})
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = factory()
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            prob = model.predict_proba(X_test_s)[:, 1]

            m = eval_metrics(y_test, pred, prob)
            m.update({"model": tag, "ratio": ratio, "skipped": False,
                        "n_test": len(y_test), "n_test_positives": n_test_pos})
            rows.append(m)
            print(f"  {tag} @ {1-ratio:.3f}/{ratio:.3f}: precision={m['precision']:.3f} "
                  f"recall={m['recall']:.3f} f1={m['f1']:.3f} pr_auc={m['pr_auc']:.3f} "
                  f"fpr={m['fpr']:.4f} fnr={m['fnr']:.4f} (n_test_pos={n_test_pos})")
    return rows


# ------------------------------------------------------------- neural ---

class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers, in_dim = [], 50
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                             dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(self.dropout(hidden[-1]))


class Autoencoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(50, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 50))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_mlp(X, y, hidden, lr, epochs=30):
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y).view(-1, 1)), batch_size=32, shuffle=True)
    model = MLP(tuple(hidden))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model


def train_lstm(X, y, hidden_size, lr, epochs=40):
    loader = DataLoader(TensorDataset(torch.tensor(X).unsqueeze(-1), torch.tensor(y).view(-1, 1)),
                         batch_size=32, shuffle=True)
    model = LSTMClassifier(hidden_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model


def train_autoencoder(X_normal, latent, lr, epochs=40):
    loader = DataLoader(TensorDataset(torch.tensor(X_normal)), batch_size=32, shuffle=True)
    model = Autoencoder(latent)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    return model


def run_neural():
    dataset = pd.read_csv("Data/ATD_sequence.csv")
    X_all = dataset.drop("LABEL", axis=1).values.astype(np.float32)
    y_all = dataset["LABEL"].values.astype(np.float32)

    mlp_params = load_best_params("mlp")
    lstm_params = load_best_params("lstm")
    ae_params = load_best_params("autoencoder")

    rows = []
    for ratio in RATIOS:
        X_r, y_r = resample_to_ratio(X_all, y_all, ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X_r, y_r, test_size=0.20, random_state=SEED, stratify=y_r
        )
        n_test_pos = int(y_test.sum())
        if n_test_pos < MIN_TEST_POSITIVES:
            for tag in ["mlp", "lstm", "autoencoder"]:
                print(f"  SKIP {tag} @ ratio={ratio}: only {n_test_pos} test-set attacks (< {MIN_TEST_POSITIVES})")
                rows.append({"model": tag, "ratio": ratio, "skipped": True,
                             "reason": f"only {n_test_pos} test-set positives"})
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        # MLP
        model = train_mlp(X_train_s, y_train, **mlp_params)
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(X_test_s))).numpy().flatten()
        pred = (prob > 0.5).astype(float)
        m = eval_metrics(y_test, pred, prob)
        m.update({"model": "mlp", "ratio": ratio, "skipped": False, "n_test": len(y_test), "n_test_positives": n_test_pos})
        rows.append(m)
        print(f"  mlp @ {1-ratio:.3f}/{ratio:.3f}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} pr_auc={m['pr_auc']:.3f} fpr={m['fpr']:.4f} fnr={m['fnr']:.4f}")

        # LSTM
        model = train_lstm(X_train_s, y_train, **lstm_params)
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(X_test_s).unsqueeze(-1))).numpy().flatten()
        pred = (prob > 0.5).astype(float)
        m = eval_metrics(y_test, pred, prob)
        m.update({"model": "lstm", "ratio": ratio, "skipped": False, "n_test": len(y_test), "n_test_positives": n_test_pos})
        rows.append(m)
        print(f"  lstm @ {1-ratio:.3f}/{ratio:.3f}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} pr_auc={m['pr_auc']:.3f} fpr={m['fpr']:.4f} fnr={m['fnr']:.4f}")

        # Autoencoder trains on normal-only data from this ratio's training split;
        # this is exactly the regime it's meant for, where positives are rare.
        X_normal = X_train_s[y_train == 0]
        X_fit, X_calib = train_test_split(X_normal, test_size=0.20, random_state=SEED)
        model = train_autoencoder(X_fit, **ae_params)
        model.eval()
        with torch.no_grad():
            calib_recon = model(torch.tensor(X_calib)).numpy()
            test_recon = model(torch.tensor(X_test_s)).numpy()
        calib_err = np.mean((X_calib - calib_recon) ** 2, axis=1)
        threshold = np.percentile(calib_err, 95)
        test_err = np.mean((X_test_s - test_recon) ** 2, axis=1)
        pred = (test_err > threshold).astype(int)
        m = eval_metrics(y_test, pred, test_err)
        m.update({"model": "autoencoder", "ratio": ratio, "skipped": False, "n_test": len(y_test), "n_test_positives": n_test_pos})
        rows.append(m)
        print(f"  autoencoder @ {1-ratio:.3f}/{ratio:.3f}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} pr_auc={m['pr_auc']:.3f} fpr={m['fpr']:.4f} fnr={m['fnr']:.4f}")

    return rows


if __name__ == "__main__":
    os.makedirs(f"{OUT}/results", exist_ok=True)

    print("=== Classical models across imbalance ratios ===")
    classical_rows = run_classical()

    print("\n=== Neural models across imbalance ratios ===")
    neural_rows = run_neural()

    df = pd.DataFrame(classical_rows + neural_rows)
    df.to_csv(f"{OUT}/results/class_imbalance_stress_test.csv", index=False)
    print(f"\nSaved {OUT}/results/class_imbalance_stress_test.csv")
