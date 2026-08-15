"""
calibration_and_latency.py

Threshold/calibration analysis and inference latency / model size /
"parameter count" benchmarking for the six tuned models produced by
cv_tuning_classical.py and cv_tuning_neural.py.

Calibration: reliability diagram (predicted probability vs. observed
frequency) and precision-recall curve per model, on the held-out test set.
The autoencoder has no calibration curve -- reconstruction error is not a
probability -- so it's included in the PR/latency tables but noted as N/A
for calibration.

Complexity proxy: PyTorch models report trainable parameter count
directly. TC-SVM/RF/XGBoost have no directly comparable notion of
"parameters", so each reports the complexity measure that actually
governs its inference cost (support vector count, total tree count /
estimators) alongside on-disk model size, which *is* comparable across
all six.
"""

import json
import time
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

import torch
import torch.nn as nn

OUT = "experiments/phase1_statistical_rigor"
os.makedirs(f"{OUT}/figures", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---- rebuild PyTorch architectures from saved best_params, for latency timing ----

class MLP(nn.Module):
    def __init__(self, hidden=(128, 64, 32)):
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
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                             dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(self.dropout(hidden[-1]))


class Autoencoder(nn.Module):
    def __init__(self, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(50, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 50))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_inference(predict_fn, n_repeats=20):
    # warm-up
    predict_fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


results = []
calib_data = {}

# ------------------------------------------------------------- classical ---

data = pd.read_csv("Data/ATD.csv")
from sklearn.model_selection import train_test_split
X_all = data[["MEAN", "RATIO"]].values
y_all = data["LABEL"].values
_, X_test_c, _, y_test_c = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

for tag, complexity_fn in [
    ("tc_svm", lambda m: {"support_vectors": int(m.n_support_.sum())}),
    ("random_forest", lambda m: {"n_estimators": m.n_estimators}),
    ("xgboost", lambda m: {"n_estimators": m.n_estimators}),
]:
    model = joblib.load(f"{OUT}/models/{tag}_tuned.joblib")
    scaler = joblib.load(f"{OUT}/models/{tag}_tuned_scaler.joblib")
    X_test_s = scaler.transform(X_test_c)

    preds_df = pd.read_csv(f"{OUT}/results/{tag}_test_predictions.csv")
    y_true, y_prob = preds_df["y_true"].values, preds_df["y_prob"].values

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    calib_data[tag] = {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    mean_t, std_t = time_inference(lambda: model.predict_proba(X_test_s))
    size_bytes = os.path.getsize(f"{OUT}/models/{tag}_tuned.joblib")

    results.append({
        "model": tag, "pr_auc": float(pr_auc),
        "inference_time_full_testset_s_mean": mean_t, "inference_time_full_testset_s_std": std_t,
        "inference_time_per_sample_ms": mean_t / len(X_test_s) * 1000,
        "model_size_bytes": size_bytes,
        "trainable_params": None,
        "complexity": complexity_fn(model),
    })
    print(f"{tag}: PR-AUC={pr_auc:.4f}  latency/sample={mean_t/len(X_test_s)*1000:.4f}ms  size={size_bytes/1024:.1f}KB")

# --------------------------------------------------------------- neural ----

dataset = pd.read_csv("Data/ATD_sequence.csv")
X_all_n = dataset.drop("LABEL", axis=1).values.astype(np.float32)
y_all_n = dataset["LABEL"].values.astype(np.float32)
_, X_test_n, _, y_test_n = train_test_split(X_all_n, y_all_n, test_size=0.20, random_state=42, stratify=y_all_n)

# MLP
params = load_json(f"{OUT}/results/mlp_best_params.json")
scaler = joblib.load(f"{OUT}/models/mlp_tuned_scaler.joblib")
X_test_s = scaler.transform(X_test_n).astype(np.float32)
model = MLP(hidden=tuple(params["hidden"]))
model.load_state_dict(torch.load(f"{OUT}/models/mlp_tuned.pth", map_location="cpu"))
model.eval()
X_tensor = torch.tensor(X_test_s)

preds_df = pd.read_csv(f"{OUT}/results/mlp_test_predictions.csv")
y_true, y_prob = preds_df["y_true"].values, preds_df["y_prob"].values
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
calib_data["mlp"] = {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}
prec, rec, _ = precision_recall_curve(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)

with torch.no_grad():
    mean_t, std_t = time_inference(lambda: torch.sigmoid(model(X_tensor)))
size_bytes = os.path.getsize(f"{OUT}/models/mlp_tuned.pth")
results.append({
    "model": "mlp", "pr_auc": float(pr_auc),
    "inference_time_full_testset_s_mean": mean_t, "inference_time_full_testset_s_std": std_t,
    "inference_time_per_sample_ms": mean_t / len(X_test_s) * 1000,
    "model_size_bytes": size_bytes, "trainable_params": count_params(model), "complexity": {},
})
print(f"mlp: PR-AUC={pr_auc:.4f}  latency/sample={mean_t/len(X_test_s)*1000:.4f}ms  "
      f"params={count_params(model)}  size={size_bytes/1024:.1f}KB")

# LSTM
params = load_json(f"{OUT}/results/lstm_best_params.json")
scaler = joblib.load(f"{OUT}/models/lstm_tuned_scaler.joblib")
X_test_s = scaler.transform(X_test_n).astype(np.float32)
model = LSTMClassifier(hidden_size=params["hidden_size"])
model.load_state_dict(torch.load(f"{OUT}/models/lstm_tuned.pth", map_location="cpu"))
model.eval()
X_tensor = torch.tensor(X_test_s).unsqueeze(-1)

preds_df = pd.read_csv(f"{OUT}/results/lstm_test_predictions.csv")
y_true, y_prob = preds_df["y_true"].values, preds_df["y_prob"].values
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
calib_data["lstm"] = {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}
prec, rec, _ = precision_recall_curve(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)

with torch.no_grad():
    mean_t, std_t = time_inference(lambda: torch.sigmoid(model(X_tensor)))
size_bytes = os.path.getsize(f"{OUT}/models/lstm_tuned.pth")
results.append({
    "model": "lstm", "pr_auc": float(pr_auc),
    "inference_time_full_testset_s_mean": mean_t, "inference_time_full_testset_s_std": std_t,
    "inference_time_per_sample_ms": mean_t / len(X_test_s) * 1000,
    "model_size_bytes": size_bytes, "trainable_params": count_params(model), "complexity": {},
})
print(f"lstm: PR-AUC={pr_auc:.4f}  latency/sample={mean_t/len(X_test_s)*1000:.4f}ms  "
      f"params={count_params(model)}  size={size_bytes/1024:.1f}KB")

# Autoencoder - no calibration curve (not a probability), still gets PR/latency/size
params = load_json(f"{OUT}/results/autoencoder_best_params.json")
scaler = joblib.load(f"{OUT}/models/autoencoder_tuned_scaler.joblib")
X_test_s = scaler.transform(X_test_n).astype(np.float32)
model = Autoencoder(latent=params["latent"])
model.load_state_dict(torch.load(f"{OUT}/models/autoencoder_tuned.pth", map_location="cpu"))
model.eval()
X_tensor = torch.tensor(X_test_s)

preds_df = pd.read_csv(f"{OUT}/results/autoencoder_test_predictions.csv")
y_true, y_prob = preds_df["y_true"].values, preds_df["y_prob"].values  # y_prob here = reconstruction error
prec, rec, _ = precision_recall_curve(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)
calib_data["autoencoder"] = None  # not applicable - documented in report

with torch.no_grad():
    mean_t, std_t = time_inference(lambda: model(X_tensor))
size_bytes = os.path.getsize(f"{OUT}/models/autoencoder_tuned.pth")
results.append({
    "model": "autoencoder", "pr_auc": float(pr_auc),
    "inference_time_full_testset_s_mean": mean_t, "inference_time_full_testset_s_std": std_t,
    "inference_time_per_sample_ms": mean_t / len(X_test_s) * 1000,
    "model_size_bytes": size_bytes, "trainable_params": count_params(model), "complexity": {},
})
print(f"autoencoder: PR-AUC={pr_auc:.4f}  latency/sample={mean_t/len(X_test_s)*1000:.4f}ms  "
      f"params={count_params(model)}  size={size_bytes/1024:.1f}KB (calibration: N/A, no probability output)")

# ---- save everything ----

pd.DataFrame(results).to_csv(f"{OUT}/results/calibration_latency_summary.csv", index=False)
with open(f"{OUT}/results/calibration_curves.json", "w") as f:
    json.dump(calib_data, f, indent=2)

# reliability diagram figure
plt.figure(figsize=(7, 7))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
for tag, c in calib_data.items():
    if c is not None:
        plt.plot(c["mean_predicted"], c["fraction_positive"], marker="o", label=tag)
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed frequency")
plt.title("Reliability Diagrams (Phase 1 tuned models)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/reliability_diagrams.png", dpi=300)
plt.close()

print(f"\nSaved: {OUT}/results/calibration_latency_summary.csv")
print(f"Saved: {OUT}/results/calibration_curves.json")
print(f"Saved: {OUT}/figures/reliability_diagrams.png")
