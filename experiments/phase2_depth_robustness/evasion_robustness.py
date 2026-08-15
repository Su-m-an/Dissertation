"""
evasion_robustness.py

LIMITATION, stated up front: this repo has no MATLAB source (matlab/ is
empty) and no confirmed list of which physical-layer parameters the
simulator exposes, so it isn't yet possible to check whether a
physically achievable attacker action maps to a specific feature-space
perturbation. Pending that, this is explicitly FEATURE-SPACE EVASION
ROBUSTNESS: how much a correctly-detected attack sample's already-
extracted features (MEAN/RATIO or E1-E50) would need to move, in
standardized feature space, before each model's decision flips. This is
NOT a claim that such a perturbation is achievable by a real eavesdropper
-- it measures decision-boundary margin, not attacker capability. If
MATLAB access is confirmed and its parameters are shared, this should be
redone by perturbing the simulator's own physical parameters and
re-extracting features, which would be a strictly stronger result.

Method: black-box random-direction search, applied identically to all
six models so the comparison across classical/tree/neural architectures
is fair even though only the neural models are actually differentiable.
For each correctly-classified attack sample: try random unit directions
in standardized feature space at increasing magnitude (0.25 SD steps up
to 5.0 SD, 25 random directions per magnitude step); record the smallest
magnitude at which any direction flips the prediction to "non-attack".
This is a conservative (upper-bound) estimate of the true minimum
adversarial distance. A white-box gradient method would find an equal
or smaller distance for the differentiable models. Reported anyway,
consistently, so relative robustness across models is still meaningful
even though the absolute numbers are conservative.
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

SEED = 42
P1M = "experiments/phase1_statistical_rigor/models"
P1R = "experiments/phase1_statistical_rigor/results"
OUT = "experiments/phase2_depth_robustness"
os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

N_ATTACK_SAMPLES = 150
MAGNITUDES = np.arange(0.25, 5.01, 0.25)  # standard deviations
DIRECTIONS_PER_MAGNITUDE = 25

rng = np.random.RandomState(SEED)


def min_evasion_distance(X_std, predict_fn, rng):
    """For each row of X_std (already correctly predicted 'attack'), find
    the smallest random-direction perturbation magnitude that flips the
    prediction. Returns array of distances (np.inf if never flipped)."""
    n, d = X_std.shape
    distances = np.full(n, np.inf)

    for mag in MAGNITUDES:
        still_unflipped = np.where(np.isinf(distances))[0]
        if len(still_unflipped) == 0:
            break
        for _ in range(DIRECTIONS_PER_MAGNITUDE):
            directions = rng.normal(size=(len(still_unflipped), d))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            perturbed = X_std[still_unflipped] + mag * directions
            preds = predict_fn(perturbed)
            flipped_now = still_unflipped[preds == 0]
            distances[flipped_now] = np.minimum(distances[flipped_now], mag)
            still_unflipped = np.where(np.isinf(distances))[0]
            if len(still_unflipped) == 0:
                break

    return distances


results = []

# ------------------------------------------------------- classical ---

data = pd.read_csv("Data/ATD.csv")
X_all = data[["MEAN", "RATIO"]].values
y_all = data["LABEL"].values
_, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)

for tag in ["tc_svm", "random_forest", "xgboost"]:
    model = joblib.load(f"{P1M}/{tag}_tuned.joblib")
    scaler = joblib.load(f"{P1M}/{tag}_tuned_scaler.joblib")
    X_test_s = scaler.transform(X_test)

    pred = model.predict(X_test_s)
    correct_attack_idx = np.where((pred == 1) & (y_test == 1))[0]
    sample_idx = rng.choice(correct_attack_idx, size=min(N_ATTACK_SAMPLES, len(correct_attack_idx)), replace=False)
    X_attacks = X_test_s[sample_idx]

    predict_fn = lambda X: model.predict(X)
    distances = min_evasion_distance(X_attacks, predict_fn, rng)

    evaded_within_2sd = float(np.mean(distances <= 2.0))
    finite = distances[np.isfinite(distances)]
    results.append({
        "model": tag, "n_samples": len(X_attacks),
        "median_min_distance_sd": float(np.median(finite)) if len(finite) else None,
        "mean_min_distance_sd": float(np.mean(finite)) if len(finite) else None,
        "pct_evaded_within_2sd": evaded_within_2sd,
        "pct_never_evaded_within_5sd": float(np.mean(np.isinf(distances))),
    })
    print(f"{tag}: median min-distance={np.median(finite) if len(finite) else float('nan'):.2f} SD, "
          f"{evaded_within_2sd*100:.1f}% evaded within 2 SD, "
          f"{np.mean(np.isinf(distances))*100:.1f}% never evaded within 5 SD")

# ---------------------------------------------------------- neural ---

class MLP(nn.Module):
    def __init__(self, hidden=(128, 64, 32)):
        super().__init__()
        layers, d = [], 50
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
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


dataset = pd.read_csv("Data/ATD_sequence.csv")
X_all_n = dataset.drop("LABEL", axis=1).values.astype(np.float32)
y_all_n = dataset["LABEL"].values.astype(np.float32)
_, X_test_n, _, y_test_n = train_test_split(X_all_n, y_all_n, test_size=0.20, random_state=SEED, stratify=y_all_n)

with open(f"{P1R}/mlp_best_params.json") as f:
    mlp_params = json.load(f)
scaler = joblib.load(f"{P1M}/mlp_tuned_scaler.joblib")
X_test_s = scaler.transform(X_test_n).astype(np.float32)
model = MLP(hidden=tuple(mlp_params["hidden"]))
model.load_state_dict(torch.load(f"{P1M}/mlp_tuned.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(X_test_s))).numpy().flatten()
pred = (probs > 0.5).astype(float)
correct_attack_idx = np.where((pred == 1) & (y_test_n == 1))[0]
sample_idx = rng.choice(correct_attack_idx, size=min(N_ATTACK_SAMPLES, len(correct_attack_idx)), replace=False)
X_attacks = X_test_s[sample_idx]


def mlp_predict(X):
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X.astype(np.float32)))).numpy().flatten()
    return (p > 0.5).astype(int)


distances = min_evasion_distance(X_attacks, mlp_predict, rng)
finite = distances[np.isfinite(distances)]
results.append({
    "model": "mlp", "n_samples": len(X_attacks),
    "median_min_distance_sd": float(np.median(finite)) if len(finite) else None,
    "mean_min_distance_sd": float(np.mean(finite)) if len(finite) else None,
    "pct_evaded_within_2sd": float(np.mean(distances <= 2.0)),
    "pct_never_evaded_within_5sd": float(np.mean(np.isinf(distances))),
})
print(f"mlp: median min-distance={np.median(finite) if len(finite) else float('nan'):.2f} SD, "
      f"{np.mean(distances<=2.0)*100:.1f}% evaded within 2 SD")

with open(f"{P1R}/lstm_best_params.json") as f:
    lstm_params = json.load(f)
scaler = joblib.load(f"{P1M}/lstm_tuned_scaler.joblib")
X_test_s = scaler.transform(X_test_n).astype(np.float32)
model = LSTMClassifier(hidden_size=lstm_params["hidden_size"])
model.load_state_dict(torch.load(f"{P1M}/lstm_tuned.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(X_test_s).unsqueeze(-1))).numpy().flatten()
pred = (probs > 0.5).astype(float)
correct_attack_idx = np.where((pred == 1) & (y_test_n == 1))[0]
sample_idx = rng.choice(correct_attack_idx, size=min(N_ATTACK_SAMPLES, len(correct_attack_idx)), replace=False)
X_attacks = X_test_s[sample_idx]


def lstm_predict(X):
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X.astype(np.float32)).unsqueeze(-1))).numpy().flatten()
    return (p > 0.5).astype(int)


distances = min_evasion_distance(X_attacks, lstm_predict, rng)
finite = distances[np.isfinite(distances)]
results.append({
    "model": "lstm", "n_samples": len(X_attacks),
    "median_min_distance_sd": float(np.median(finite)) if len(finite) else None,
    "mean_min_distance_sd": float(np.mean(finite)) if len(finite) else None,
    "pct_evaded_within_2sd": float(np.mean(distances <= 2.0)),
    "pct_never_evaded_within_5sd": float(np.mean(np.isinf(distances))),
})
print(f"lstm: median min-distance={np.median(finite) if len(finite) else float('nan'):.2f} SD, "
      f"{np.mean(distances<=2.0)*100:.1f}% evaded within 2 SD")

# Autoencoder: "evasion" = pushing reconstruction error below the threshold
with open(f"{P1R}/autoencoder_best_params.json") as f:
    ae_params = json.load(f)
with open(f"{P1M}/autoencoder_tuned_threshold.json") as f:
    threshold = json.load(f)["threshold"]
scaler = joblib.load(f"{P1M}/autoencoder_tuned_scaler.joblib")
X_test_s = scaler.transform(X_test_n).astype(np.float32)
model = Autoencoder(latent=ae_params["latent"])
model.load_state_dict(torch.load(f"{P1M}/autoencoder_tuned.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    recon = model(torch.tensor(X_test_s)).numpy()
errors = np.mean((X_test_s - recon) ** 2, axis=1)
pred = (errors > threshold).astype(float)
correct_attack_idx = np.where((pred == 1) & (y_test_n == 1))[0]
sample_idx = rng.choice(correct_attack_idx, size=min(N_ATTACK_SAMPLES, len(correct_attack_idx)), replace=False)
X_attacks = X_test_s[sample_idx]


def ae_predict(X):
    with torch.no_grad():
        r = model(torch.tensor(X.astype(np.float32))).numpy()
    e = np.mean((X - r) ** 2, axis=1)
    return (e > threshold).astype(int)


distances = min_evasion_distance(X_attacks, ae_predict, rng)
finite = distances[np.isfinite(distances)]
results.append({
    "model": "autoencoder", "n_samples": len(X_attacks),
    "median_min_distance_sd": float(np.median(finite)) if len(finite) else None,
    "mean_min_distance_sd": float(np.mean(finite)) if len(finite) else None,
    "pct_evaded_within_2sd": float(np.mean(distances <= 2.0)),
    "pct_never_evaded_within_5sd": float(np.mean(np.isinf(distances))),
})
print(f"autoencoder: median min-distance={np.median(finite) if len(finite) else float('nan'):.2f} SD, "
      f"{np.mean(distances<=2.0)*100:.1f}% evaded within 2 SD")

df = pd.DataFrame(results)
df.to_csv(f"{OUT}/results/evasion_robustness_feature_space.csv", index=False)

plt.figure(figsize=(9, 6))
plt.bar(df["model"], df["median_min_distance_sd"])
plt.ylabel("Median minimum evasion distance (standard deviations)")
plt.title("Feature-space evasion robustness (higher = harder to evade)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/evasion_robustness.png", dpi=300)
plt.close()

print(f"\nSaved {OUT}/results/evasion_robustness_feature_space.csv")
print(f"Saved {OUT}/figures/evasion_robustness.png")
