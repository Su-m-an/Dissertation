"""
cv_tuning_neural.py

Phase 1 statistical rigor for the sequence-based models (MLP, LSTM,
Autoencoder), trained on the 50-step ATD_sequence.csv. It only has 4,000
rows, so unlike TC-SVM there is no cost reason to subsample: full-scale
grid search plus full-scale 5-fold CV throughout.

MLP and LSTM: standard supervised CV, where accuracy is the search/report
metric. Autoencoder: unsupervised on normal-only training data, so the CV
score is reconstruction-error ROC-AUC (threshold-free) during the search.
The full CV report additionally computes threshold-based metrics using
the same normal-only threshold calibration as the corrected baseline
(src/12_autoencoder.py), calibrated on held-out normal rows from each
fold's own training portion and never on the validation fold.

Writes only under experiments/phase1_statistical_rigor/.
"""

import json
import time
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
K = 5
OUT = "experiments/phase1_statistical_rigor"

os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/models", exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

dataset = pd.read_csv("Data/ATD_sequence.csv")
X_all = dataset.drop("LABEL", axis=1).values.astype(np.float32)
y_all = dataset["LABEL"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)


# ---------------------------------------------------------------- MLP ----

class MLP(nn.Module):
    def __init__(self, hidden=(128, 64, 32)):
        super().__init__()
        layers = []
        in_dim = 50
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, hidden, lr, epochs=30, seed=SEED):
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).view(-1, 1)),
        batch_size=32, shuffle=True
    )
    model = MLP(hidden)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model


def eval_mlp(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_val))).numpy().flatten()
    preds = (probs > 0.5).astype(float)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "f1": f1_score(y_val, preds),
        "auc": roc_auc_score(y_val, probs),
    }


# --------------------------------------------------------------- LSTM ----

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                             dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(self.dropout(hidden[-1]))


def train_lstm(X_tr, y_tr, hidden_size, lr, epochs=40, seed=SEED):
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr).unsqueeze(-1), torch.tensor(y_tr).view(-1, 1)),
        batch_size=32, shuffle=True
    )
    model = LSTMClassifier(hidden_size=hidden_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model


def eval_lstm(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_val).unsqueeze(-1))).numpy().flatten()
    preds = (probs > 0.5).astype(float)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "f1": f1_score(y_val, preds),
        "auc": roc_auc_score(y_val, probs),
    }


# ---------------------------------------------------------- Autoencoder ---

class Autoencoder(nn.Module):
    def __init__(self, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(50, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 50))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_normal_tr, latent, lr, epochs=40, seed=SEED):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_normal_tr)), batch_size=32, shuffle=True)
    model = Autoencoder(latent=latent)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    return model


def ae_errors(model, X):
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X)).numpy()
    return np.mean((X - recon) ** 2, axis=1)


# ------------------------------------------------------------ CV driver ---

def stratified_folds(X, y, k=K, seed=SEED):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def run_mlp():
    tag = "mlp"
    print(f"\n=== {tag} ===")
    t0 = time.time()
    grid = {"hidden": [(128, 64, 32), (64, 32)], "lr": [0.0005, 0.001, 0.002]}

    folds = stratified_folds(X_train, y_train)
    search_results, best_params, best_score = [], None, -np.inf

    for params in ParameterGrid(grid):
        accs = []
        for train_idx, val_idx in folds:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train[train_idx]).astype(np.float32)
            X_val = scaler.transform(X_train[val_idx]).astype(np.float32)
            model = train_mlp(X_tr, y_train[train_idx], **params)
            m = eval_mlp(model, X_val, y_train[val_idx])
            accs.append(m["accuracy"])
        score = float(np.mean(accs))
        search_results.append({"hidden": str(params["hidden"]), "lr": params["lr"], "cv_accuracy": score})
        print(f"    params={params} -> cv_acc={score:.4f}")
        if score > best_score:
            best_score, best_params = score, params

    search_time = time.time() - t0
    print(f"  Best: {best_params} (cv_acc={best_score:.4f}), search_time={search_time:.1f}s")

    with open(f"{OUT}/results/{tag}_search_grid.json", "w") as f:
        json.dump({"grid": {"hidden": [list(h) for h in grid["hidden"]], "lr": grid["lr"]},
                    "results": search_results, "search_time_s": search_time}, f, indent=2)
    with open(f"{OUT}/results/{tag}_best_params.json", "w") as f:
        json.dump({"hidden": list(best_params["hidden"]), "lr": best_params["lr"]}, f, indent=2)

    t1 = time.time()
    rows = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        assert len(set(train_idx) & set(val_idx)) == 0
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[train_idx]).astype(np.float32)
        X_val = scaler.transform(X_train[val_idx]).astype(np.float32)
        model = train_mlp(X_tr, y_train[train_idx], **best_params)
        m = eval_mlp(model, X_val, y_train[val_idx])
        m["fold"] = i
        rows.append(m)
        print(f"    [{tag} fold {i}/{K}] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    cv_time = time.time() - t1
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(f"{OUT}/results/{tag}_cv_folds.csv", index=False)
    print(f"  CV accuracy: {cv_df['accuracy'].mean():.4f} +/- {cv_df['accuracy'].std():.4f} (time={cv_time:.1f}s)")

    t2 = time.time()
    scaler = StandardScaler()
    X_tr_full = scaler.fit_transform(X_train).astype(np.float32)
    X_te_full = scaler.transform(X_test).astype(np.float32)
    final_model = train_mlp(X_tr_full, y_train, **best_params)
    refit_time = time.time() - t2
    final_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(final_model(torch.tensor(X_te_full))).numpy().flatten()
    preds = (probs > 0.5).astype(float)

    torch.save(final_model.state_dict(), f"{OUT}/models/{tag}_tuned.pth")
    joblib.dump(scaler, f"{OUT}/models/{tag}_tuned_scaler.joblib")
    pd.DataFrame({"y_true": y_test, "y_pred": preds, "y_prob": probs}).to_csv(
        f"{OUT}/results/{tag}_test_predictions.csv", index=False)

    total_time = time.time() - t0
    with open(f"{OUT}/results/{tag}_timing.json", "w") as f:
        json.dump({"search_time_s": search_time, "cv_time_s": cv_time,
                    "final_refit_time_s": refit_time, "total_time_s": total_time}, f, indent=2)
    print(f"  {tag} done in {total_time:.1f}s total.")


def run_lstm():
    tag = "lstm"
    print(f"\n=== {tag} ===")
    t0 = time.time()
    grid = {"hidden_size": [32, 64, 128], "lr": [0.0005, 0.001]}

    folds = stratified_folds(X_train, y_train)
    search_results, best_params, best_score = [], None, -np.inf

    for params in ParameterGrid(grid):
        accs = []
        for train_idx, val_idx in folds:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train[train_idx]).astype(np.float32)
            X_val = scaler.transform(X_train[val_idx]).astype(np.float32)
            model = train_lstm(X_tr, y_train[train_idx], **params)
            m = eval_lstm(model, X_val, y_train[val_idx])
            accs.append(m["accuracy"])
        score = float(np.mean(accs))
        search_results.append({**params, "cv_accuracy": score})
        print(f"    params={params} -> cv_acc={score:.4f}")
        if score > best_score:
            best_score, best_params = score, params

    search_time = time.time() - t0
    print(f"  Best: {best_params} (cv_acc={best_score:.4f}), search_time={search_time:.1f}s")

    with open(f"{OUT}/results/{tag}_search_grid.json", "w") as f:
        json.dump({"grid": grid, "results": search_results, "search_time_s": search_time}, f, indent=2)
    with open(f"{OUT}/results/{tag}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    t1 = time.time()
    rows = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        assert len(set(train_idx) & set(val_idx)) == 0
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[train_idx]).astype(np.float32)
        X_val = scaler.transform(X_train[val_idx]).astype(np.float32)
        model = train_lstm(X_tr, y_train[train_idx], **best_params)
        m = eval_lstm(model, X_val, y_train[val_idx])
        m["fold"] = i
        rows.append(m)
        print(f"    [{tag} fold {i}/{K}] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    cv_time = time.time() - t1
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(f"{OUT}/results/{tag}_cv_folds.csv", index=False)
    print(f"  CV accuracy: {cv_df['accuracy'].mean():.4f} +/- {cv_df['accuracy'].std():.4f} (time={cv_time:.1f}s)")

    t2 = time.time()
    scaler = StandardScaler()
    X_tr_full = scaler.fit_transform(X_train).astype(np.float32)
    X_te_full = scaler.transform(X_test).astype(np.float32)
    final_model = train_lstm(X_tr_full, y_train, **best_params)
    refit_time = time.time() - t2
    final_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(final_model(torch.tensor(X_te_full).unsqueeze(-1))).numpy().flatten()
    preds = (probs > 0.5).astype(float)

    torch.save(final_model.state_dict(), f"{OUT}/models/{tag}_tuned.pth")
    joblib.dump(scaler, f"{OUT}/models/{tag}_tuned_scaler.joblib")
    pd.DataFrame({"y_true": y_test, "y_pred": preds, "y_prob": probs}).to_csv(
        f"{OUT}/results/{tag}_test_predictions.csv", index=False)

    total_time = time.time() - t0
    with open(f"{OUT}/results/{tag}_timing.json", "w") as f:
        json.dump({"search_time_s": search_time, "cv_time_s": cv_time,
                    "final_refit_time_s": refit_time, "total_time_s": total_time}, f, indent=2)
    print(f"  {tag} done in {total_time:.1f}s total.")


def run_autoencoder():
    tag = "autoencoder"
    print(f"\n=== {tag} ===")
    t0 = time.time()
    grid = {"latent": [4, 8, 16], "lr": [0.0005, 0.001]}

    folds = stratified_folds(X_train, y_train)
    search_results, best_params, best_score = [], None, -np.inf

    for params in ParameterGrid(grid):
        aucs = []
        for train_idx, val_idx in folds:
            fold_X_tr, fold_y_tr = X_train[train_idx], y_train[train_idx]
            X_normal = fold_X_tr[fold_y_tr == 0]
            scaler = StandardScaler()
            X_normal_s = scaler.fit_transform(X_normal).astype(np.float32)
            X_val_s = scaler.transform(X_train[val_idx]).astype(np.float32)
            model = train_autoencoder(X_normal_s, **params)
            errors = ae_errors(model, X_val_s)
            aucs.append(roc_auc_score(y_train[val_idx], errors))
        score = float(np.mean(aucs))
        search_results.append({**params, "cv_auc": score})
        print(f"    params={params} -> cv_auc={score:.4f}")
        if score > best_score:
            best_score, best_params = score, params

    search_time = time.time() - t0
    print(f"  Best: {best_params} (cv_auc={best_score:.4f}), search_time={search_time:.1f}s")

    with open(f"{OUT}/results/{tag}_search_grid.json", "w") as f:
        json.dump({"grid": grid, "results": search_results, "search_time_s": search_time}, f, indent=2)
    with open(f"{OUT}/results/{tag}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    t1 = time.time()
    rows = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        assert len(set(train_idx) & set(val_idx)) == 0
        fold_X_tr, fold_y_tr = X_train[train_idx], y_train[train_idx]
        X_normal = fold_X_tr[fold_y_tr == 0]
        # threshold calibrated on held-out normal rows from this fold's training portion only
        X_fit, X_calib = train_test_split(X_normal, test_size=0.20, random_state=SEED)
        scaler = StandardScaler()
        X_fit_s = scaler.fit_transform(X_fit).astype(np.float32)
        X_calib_s = scaler.transform(X_calib).astype(np.float32)
        X_val_s = scaler.transform(X_train[val_idx]).astype(np.float32)

        model = train_autoencoder(X_fit_s, **best_params)
        calib_errors = ae_errors(model, X_calib_s)
        threshold = np.percentile(calib_errors, 95)
        val_errors = ae_errors(model, X_val_s)
        preds = (val_errors > threshold).astype(int)

        y_val = y_train[val_idx]
        m = {
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds),
            "recall": recall_score(y_val, preds),
            "f1": f1_score(y_val, preds),
            "auc": roc_auc_score(y_val, val_errors),
            "fold": i,
        }
        rows.append(m)
        print(f"    [{tag} fold {i}/{K}] acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")
    cv_time = time.time() - t1
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(f"{OUT}/results/{tag}_cv_folds.csv", index=False)
    print(f"  CV accuracy: {cv_df['accuracy'].mean():.4f} +/- {cv_df['accuracy'].std():.4f} (time={cv_time:.1f}s)")

    # Final refit: normal-only fit + calib split from full training data, evaluate on test
    t2 = time.time()
    X_normal_full = X_train[y_train == 0]
    X_fit_full, X_calib_full = train_test_split(X_normal_full, test_size=0.20, random_state=SEED)
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit_full).astype(np.float32)
    X_calib_s = scaler.transform(X_calib_full).astype(np.float32)
    X_te_s = scaler.transform(X_test).astype(np.float32)
    final_model = train_autoencoder(X_fit_s, **best_params)
    refit_time = time.time() - t2

    calib_errors = ae_errors(final_model, X_calib_s)
    threshold = np.percentile(calib_errors, 95)
    test_errors = ae_errors(final_model, X_te_s)
    preds = (test_errors > threshold).astype(int)

    torch.save(final_model.state_dict(), f"{OUT}/models/{tag}_tuned.pth")
    joblib.dump(scaler, f"{OUT}/models/{tag}_tuned_scaler.joblib")
    with open(f"{OUT}/models/{tag}_tuned_threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    pd.DataFrame({"y_true": y_test, "y_pred": preds, "y_prob": test_errors}).to_csv(
        f"{OUT}/results/{tag}_test_predictions.csv", index=False)

    total_time = time.time() - t0
    with open(f"{OUT}/results/{tag}_timing.json", "w") as f:
        json.dump({"search_time_s": search_time, "cv_time_s": cv_time,
                    "final_refit_time_s": refit_time, "total_time_s": total_time}, f, indent=2)
    print(f"  {tag} done in {total_time:.1f}s total.")


if __name__ == "__main__":
    overall_start = time.time()
    run_mlp()
    run_lstm()
    run_autoencoder()
    print(f"\nAll neural models complete in {time.time() - overall_start:.1f}s total.")
