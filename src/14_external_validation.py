"""
14_external_validation.py

Data/ATD_features.csv is a second, independently generated MATLAB
simulation run (same MEAN/RATIO feature extraction, same class balance,
but no overlapping rows with Data/ATD.csv). It was never used anywhere
in the pipeline, so this script wires it in as a genuine external
generalisation test: the classical models trained in 06/07/08 (loaded
from saved_models/, together with the scaler each was fit with on
ATD.csv) are evaluated on it, unseen and untuned.

Only the classical models apply here (TC-SVM, Random Forest, XGBoost) -
they're the ones trained on the MEAN/RATIO features that ATD_features.csv
also provides. The deep models operate on ATD_sequence.csv's 50-step
sequences, which has no external counterpart.
"""

import joblib

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

models = {
    "TC-SVM": (
        "saved_models/tc_svm.joblib",
        "saved_models/tc_svm_scaler.joblib"
    ),
    "Random Forest": (
        "saved_models/random_forest.joblib",
        "saved_models/random_forest_scaler.joblib"
    ),
    "XGBoost": (
        "saved_models/xgboost.joblib",
        "saved_models/xgboost_scaler.joblib"
    )
}

# Load the external simulation run

external = pd.read_csv("Data/ATD_features.csv")

X_external = external[["MEAN", "RATIO"]]
y_external = external["LABEL"]

print(f"\nExternal validation set: {X_external.shape[0]} samples\n")

rows = []

for name, (model_path, scaler_path) in models.items():

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X_external)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    accuracy = accuracy_score(y_external, y_pred)
    precision = precision_score(y_external, y_pred)
    recall = recall_score(y_external, y_pred)
    f1 = f1_score(y_external, y_pred)
    auc = roc_auc_score(y_external, y_prob)

    print(f"{name}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}\n")

    rows.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    })

comparison = pd.DataFrame(rows).round(4)

comparison.to_csv(
    "results/external_validation_metrics.csv",
    index=False
)

# Compare against the original held-out test split (results/model_comparison.csv)
# to show how much accuracy each model retains on an unseen simulation run.

internal = pd.read_csv("results/model_comparison.csv")

merged = internal.merge(
    comparison,
    on="Model",
    suffixes=(" (Internal Test)", " (External Validation)")
)

plt.figure(figsize=(9, 6))

x = range(len(merged))

plt.bar(
    [i - 0.2 for i in x],
    merged["Accuracy (Internal Test)"],
    width=0.4,
    label="Internal Test Split"
)

plt.bar(
    [i + 0.2 for i in x],
    merged["Accuracy (External Validation)"],
    width=0.4,
    label="External Validation Set"
)

plt.xticks(list(x), merged["Model"])
plt.ylabel("Accuracy")
plt.title("Internal Test vs External Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/external_validation_comparison.png",
    dpi=300
)

plt.close()

print("External validation metrics saved to results/external_validation_metrics.csv")
print("Comparison figure saved to figures/external_validation_comparison.png")
