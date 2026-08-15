"""
external_validation_replicates.py

Extends the original single external-validation run (src/14_external_validation.py,
Data/ATD_features.csv) to five independent replicates, generated at the
same baseline parameters (rho_u=rho_E=5) but distinct explicit seeds
(43-47) via generate_sweep_data.sh, saved to raw_external/seed_*/ATD_features.csv.

Uses the Phase 1 tuned classical models directly and unmodified, exactly
as the original external validation did, so there's no retraining here.
This measures generalization of the already-fixed models across
independent simulation draws, reporting mean +/- SD across replicates
instead of a single point estimate.
"""

import glob
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

P1M = "experiments/phase1_statistical_rigor/models"
OUT = "experiments/phase3_physical_layer"

os.makedirs(f"{OUT}/results", exist_ok=True)

replicate_dirs = sorted(glob.glob(f"{OUT}/raw_external/seed_*"))
print(f"Found {len(replicate_dirs)} external replicates: {replicate_dirs}")

rows = []

for tag in ["tc_svm", "random_forest", "xgboost"]:
    model = joblib.load(f"{P1M}/{tag}_tuned.joblib")
    scaler = joblib.load(f"{P1M}/{tag}_tuned_scaler.joblib")

    for d in replicate_dirs:
        seed = d.split("seed_")[1]
        external = pd.read_csv(f"{d}/ATD_features.csv")
        X = external[["MEAN", "RATIO"]].values
        y = external["LABEL"].values

        X_s = scaler.transform(X)
        pred = model.predict(X_s)
        prob = model.predict_proba(X_s)[:, 1]

        rows.append({
            "model": tag, "seed": seed,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, prob),
        })

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/results/external_validation_replicates.csv", index=False)

print("\n=== Per-replicate results ===")
print(df.to_string(index=False))

summary = df.groupby("model")[["accuracy", "precision", "recall", "f1", "auc"]].agg(["mean", "std"])
summary.to_csv(f"{OUT}/results/external_validation_replicates_summary.csv")

print("\n=== Summary across replicates (mean +/- SD) ===")
for tag in ["tc_svm", "random_forest", "xgboost"]:
    sub = df[df["model"] == tag]
    print(f"{tag}: accuracy={sub['accuracy'].mean():.4f} +/- {sub['accuracy'].std():.4f}  "
          f"(n={len(sub)} replicates)")

print(f"\nSaved {OUT}/results/external_validation_replicates.csv")
print(f"Saved {OUT}/results/external_validation_replicates_summary.csv")
