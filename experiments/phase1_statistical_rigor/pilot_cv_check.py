"""
pilot_cv_check.py

Small-scale, fast validation of the stratified k-fold CV harness before
committing to the full Phase 1 experiment suite. Verifies, on a subsample:

  1. No index leakage between train and validation folds.
  2. Preprocessing (StandardScaler) is fit only on each fold's training
     data, never on validation data or the full dataset up front.
  3. Two runs with the same seed produce identical fold assignments and
     identical metrics (reproducibility).
  4. Outputs are written under experiments/, not results/ (baseline stays
     untouched).

Run on a stratified subsample (3,000 rows for the classical/tabular
representation, the full 4,000-row sequence set) with k=3 folds, using
one representative classical model (TC-SVM) and one representative
sequence model (MLP) - not the full six-model x k-fold x hyperparameter
grid, which is the expensive part deferred to the full run.
"""

import json
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
K = 3
OUT_DIR = "experiments/phase1_statistical_rigor/pilot"

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def check_no_leakage(train_idx, val_idx, fold_num, tag):
    overlap = set(train_idx) & set(val_idx)
    assert len(overlap) == 0, f"LEAKAGE: {tag} fold {fold_num} has {len(overlap)} overlapping indices"
    log(f"  [{tag} fold {fold_num}] train/val index overlap: {len(overlap)} (OK)")


def run_svm_pilot():
    log("\n=== TC-SVM pilot (classical / ATD.csv) ===")

    data = pd.read_csv("Data/ATD.csv")

    # Stratified subsample for the pilot only - full-scale run uses all 184,000 rows
    _, subsample = train_test_split(
        data, test_size=3000, random_state=SEED, stratify=data["LABEL"]
    )
    X = subsample[["MEAN", "RATIO"]].values
    y = subsample["LABEL"].values

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    fold_metrics = []
    scaler_means = []

    start = time.time()

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        check_no_leakage(train_idx, val_idx, fold_num, "SVM")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scaler fit strictly on this fold's training data
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        scaler_means.append(scaler.mean_.copy())

        model = SVC(kernel="rbf", C=1, gamma=0.001, probability=False, random_state=SEED)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_val_s)

        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        fold_metrics.append({"fold": fold_num, "accuracy": acc, "f1": f1})
        log(f"  [SVM fold {fold_num}] acc={acc:.4f} f1={f1:.4f} (scaler mean={scaler.mean_})")

    elapsed = time.time() - start

    # Confirm scaler was actually refit per fold (means should differ across folds,
    # proving it wasn't fit once globally and reused)
    distinct_means = len(set(tuple(np.round(m, 6)) for m in scaler_means))
    log(f"  Distinct per-fold scaler means: {distinct_means} / {K} (>{1} confirms per-fold refit)")
    assert distinct_means > 1, "LEAKAGE SUSPECTED: scaler means identical across folds"

    log(f"  SVM pilot wall time ({K} folds, n=3000): {elapsed:.2f}s")

    return fold_metrics, elapsed


def run_mlp_pilot():
    log("\n=== MLP pilot (sequence / ATD_sequence.csv) ===")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = pd.read_csv("Data/ATD_sequence.csv")
    X = dataset.drop("LABEL", axis=1).values.astype(np.float32)
    y = dataset["LABEL"].values.astype(np.float32)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    fold_metrics = []
    scaler_means = []

    start = time.time()

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        check_no_leakage(train_idx, val_idx, fold_num, "MLP")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        scaler_means.append(scaler.mean_.copy())

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train).view(-1, 1)),
            batch_size=32, shuffle=True
        )

        model = nn.Sequential(
            nn.Linear(50, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):  # short pilot run, not the full 30 epochs
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X_val_s)))
            pred = (probs > 0.5).float().numpy().flatten()

        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        fold_metrics.append({"fold": fold_num, "accuracy": acc, "f1": f1})
        log(f"  [MLP fold {fold_num}] acc={acc:.4f} f1={f1:.4f}")

    elapsed = time.time() - start

    distinct_means = len(set(tuple(np.round(m, 6)) for m in scaler_means))
    log(f"  Distinct per-fold scaler means: {distinct_means} / {K}")
    assert distinct_means > 1, "LEAKAGE SUSPECTED: scaler means identical across folds"

    log(f"  MLP pilot wall time ({K} folds, n=4000, 10 epochs): {elapsed:.2f}s")

    return fold_metrics, elapsed


def check_reproducibility(run_fn, tag):
    log(f"\n=== Reproducibility check: {tag} (same seed, two runs) ===")
    m1, _ = run_fn()
    m2, _ = run_fn()
    identical = all(
        abs(a["accuracy"] - b["accuracy"]) < 1e-9 and abs(a["f1"] - b["f1"]) < 1e-9
        for a, b in zip(m1, m2)
    )
    log(f"  Run 1 == Run 2 (bitwise metric match): {identical}")
    assert identical, f"REPRODUCIBILITY FAILURE: {tag} produced different results with the same seed"
    return m1


if __name__ == "__main__":

    log(f"Phase 1 pilot - stratified {K}-fold CV harness validation")
    log(f"Seed: {SEED}")

    svm_metrics = check_reproducibility(run_svm_pilot, "TC-SVM")
    mlp_metrics = check_reproducibility(run_mlp_pilot, "MLP")

    svm_acc = [m["accuracy"] for m in svm_metrics]
    mlp_acc = [m["accuracy"] for m in mlp_metrics]

    log("\n=== Summary ===")
    log(f"SVM pilot accuracy: {np.mean(svm_acc):.4f} +/- {np.std(svm_acc):.4f} (n=3000, k={K})")
    log(f"MLP pilot accuracy: {np.mean(mlp_acc):.4f} +/- {np.std(mlp_acc):.4f} (n=4000, k={K})")
    log("\nAll leakage, refit, and reproducibility checks PASSED.")

    result = {
        "seed": SEED,
        "k_folds": K,
        "svm_pilot": svm_metrics,
        "mlp_pilot": mlp_metrics,
        "checks_passed": [
            "no_train_val_index_overlap",
            "per_fold_scaler_refit_confirmed",
            "reproducible_across_repeated_runs"
        ]
    }

    with open(f"{OUT_DIR}/pilot_results.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(f"{OUT_DIR}/pilot_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nPilot artifacts written to {OUT_DIR}/")
