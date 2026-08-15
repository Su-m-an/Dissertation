"""
error_analysis.py

Characterizes *where* each model gets it wrong, using the Phase 1 tuned
models' saved test-set predictions (experiments/phase1_statistical_rigor/
results/*_test_predictions.csv) -- no retraining needed here.

Classical models (TC-SVM/RF/XGBoost): misclassified points plotted in the
original MEAN/RATIO feature space -- are errors concentrated near the
decision boundary, or scattered?

Sequence models (MLP/LSTM): predicted-probability distributions for
correct vs. incorrect predictions -- are errors confidently wrong, or
boundary cases sitting near 0.5?

Autoencoder: reconstruction-error distribution for false negatives
(missed attacks) vs. true positives -- do missed attacks look like
"nearly normal" sequences that fall just under the threshold?
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

SEED = 42
P1 = "experiments/phase1_statistical_rigor/results"
OUT = "experiments/phase2_depth_robustness"

os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

summary_rows = []

# ------------------------------------------------------- classical ---

data = pd.read_csv("Data/ATD.csv")
X_all = data[["MEAN", "RATIO"]].values
y_all = data["LABEL"].values
_, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, tag in zip(axes, ["tc_svm", "random_forest", "xgboost"]):
    preds = pd.read_csv(f"{P1}/{tag}_test_predictions.csv")
    assert len(preds) == len(X_test)
    correct = preds["y_true"].values == preds["y_pred"].values

    ax.scatter(X_test[correct, 0], X_test[correct, 1], s=3, alpha=0.15, color="gray", label="Correct")
    ax.scatter(X_test[~correct, 0], X_test[~correct, 1], s=8, alpha=0.7, color="crimson", label="Misclassified")
    ax.set_title(f"{tag} ({(~correct).sum()} errors / {len(correct)})")
    ax.set_xlabel("MEAN")
    ax.set_ylabel("RATIO")
    ax.legend()

    fp = int(((preds["y_pred"] == 1) & (preds["y_true"] == 0)).sum())
    fn = int(((preds["y_pred"] == 0) & (preds["y_true"] == 1)).sum())
    summary_rows.append({"model": tag, "n_errors": int((~correct).sum()), "false_positives": fp, "false_negatives": fn})
    print(f"{tag}: {(~correct).sum()} errors ({fp} FP, {fn} FN) out of {len(correct)}")

plt.tight_layout()
plt.savefig(f"{OUT}/figures/error_analysis_classical_scatter.png", dpi=300)
plt.close()

# ---------------------------------------------------------- neural ---

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, tag in zip(axes, ["mlp", "lstm"]):
    preds = pd.read_csv(f"{P1}/{tag}_test_predictions.csv")
    correct = preds["y_true"].values == preds["y_pred"].values

    ax.hist(preds.loc[correct, "y_prob"], bins=40, alpha=0.6, label="Correct", color="steelblue")
    ax.hist(preds.loc[~correct, "y_prob"], bins=40, alpha=0.8, label="Misclassified", color="crimson")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{tag} predicted-probability distribution")
    ax.set_xlabel("Predicted P(attack)")
    ax.set_ylabel("Count")
    ax.legend()

    # how many errors are "confident" (prob < 0.1 or > 0.9) vs boundary cases?
    err_probs = preds.loc[~correct, "y_prob"]
    boundary = int(((err_probs > 0.3) & (err_probs < 0.7)).sum())
    confident_wrong = int(len(err_probs) - boundary)
    fp = int(((preds["y_pred"] == 1) & (preds["y_true"] == 0)).sum())
    fn = int(((preds["y_pred"] == 0) & (preds["y_true"] == 1)).sum())
    summary_rows.append({"model": tag, "n_errors": int((~correct).sum()), "false_positives": fp, "false_negatives": fn,
                            "boundary_errors_0.3to0.7": boundary, "confident_wrong_errors": confident_wrong})
    print(f"{tag}: {(~correct).sum()} errors ({fp} FP, {fn} FN); {boundary} near-boundary (0.3-0.7), "
          f"{confident_wrong} confidently wrong")

plt.tight_layout()
plt.savefig(f"{OUT}/figures/error_analysis_neural_probability.png", dpi=300)
plt.close()

# ------------------------------------------------------ autoencoder ---

preds = pd.read_csv(f"{P1}/autoencoder_test_predictions.csv")
tp_mask = (preds["y_pred"] == 1) & (preds["y_true"] == 1)
fn_mask = (preds["y_pred"] == 0) & (preds["y_true"] == 1)
fp_mask = (preds["y_pred"] == 1) & (preds["y_true"] == 0)

plt.figure(figsize=(8, 6))
plt.hist(preds.loc[tp_mask, "y_prob"], bins=30, alpha=0.6, label=f"True positives (n={tp_mask.sum()})", color="steelblue")
plt.hist(preds.loc[fn_mask, "y_prob"], bins=30, alpha=0.8, label=f"False negatives (n={fn_mask.sum()})", color="crimson")
plt.xlabel("Reconstruction error")
plt.ylabel("Count")
plt.title("Autoencoder: missed attacks vs. detected attacks")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/figures/error_analysis_autoencoder_reconstruction.png", dpi=300)
plt.close()

fn_err = preds.loc[fn_mask, "y_prob"]
tp_err = preds.loc[tp_mask, "y_prob"]
summary_rows.append({
    "model": "autoencoder", "n_errors": int(fn_mask.sum() + fp_mask.sum()),
    "false_positives": int(fp_mask.sum()), "false_negatives": int(fn_mask.sum()),
    "fn_mean_recon_error": float(fn_err.mean()) if len(fn_err) else None,
    "tp_mean_recon_error": float(tp_err.mean()) if len(tp_err) else None,
})
print(f"autoencoder: {fn_mask.sum()} FN (mean recon error {fn_err.mean():.4f}) vs "
      f"{tp_mask.sum()} TP (mean recon error {tp_err.mean():.4f})")

pd.DataFrame(summary_rows).to_csv(f"{OUT}/results/error_analysis_summary.csv", index=False)
print(f"\nSaved {OUT}/results/error_analysis_summary.csv")
print("Figures: error_analysis_classical_scatter.png, error_analysis_neural_probability.png, "
      "error_analysis_autoencoder_reconstruction.png")
