"""
interpretability.py

Uses the Phase 1 tuned models (experiments/phase1_statistical_rigor/models/)
-- no retraining. Three different interpretive vocabularies, deliberately
NOT interchangeable across model types:

  Random Forest / XGBoost -- SHAP values (shap.TreeExplainer). These are
  tree models over 2 features (MEAN, RATIO), so this mainly confirms/
  extends the built-in feature-importance figures already in results/
  with per-prediction attribution and directionality.

  LSTM -- has no reconstruction target, so reconstruction-error language
  does not apply. Instead: (1) timestep occlusion (replace each timestep
  with the sequence mean, one at a time, measure the drop in predicted
  probability of the true class), (2) timestep permutation importance
  (shuffle one timestep's values across the batch, measure the accuracy
  drop), (3) Integrated Gradients via captum, attributing the prediction
  back to each of the 50 input timesteps against a zero baseline.

  Autoencoder -- its native vocabulary IS reconstruction error, used
  correctly here: per-timestep reconstruction error, decomposed for
  attack vs. normal test samples, to see which timesteps the model
  reconstructs worst when an attack is present.
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

import shap
from captum.attr import IntegratedGradients

SEED = 42
P1M = "experiments/phase1_statistical_rigor/models"
P1R = "experiments/phase1_statistical_rigor/results"
OUT = "experiments/phase2_depth_robustness"
os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------ RF / XGBoost (SHAP) ---

def run_tree_shap():
    data = pd.read_csv("Data/ATD.csv")
    X_all = data[["MEAN", "RATIO"]].values
    y_all = data["LABEL"].values
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)

    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(X_test), size=2000, replace=False)
    X_sample = X_test[sample_idx]

    for tag in ["random_forest", "xgboost"]:
        model = joblib.load(f"{P1M}/{tag}_tuned.joblib")
        scaler = joblib.load(f"{P1M}/{tag}_tuned_scaler.joblib")
        X_sample_s = scaler.transform(X_sample)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample_s)
        if isinstance(sv, list):  # some sklearn/xgboost versions return per-class list
            sv = sv[1]
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, 1]

        plt.figure(figsize=(8, 5))
        shap.summary_plot(sv, X_sample_s, feature_names=["MEAN", "RATIO"], show=False)
        plt.title(f"{tag} SHAP summary")
        plt.tight_layout()
        plt.savefig(f"{OUT}/figures/shap_summary_{tag}.png", dpi=300)
        plt.close()

        mean_abs_shap = np.abs(sv).mean(axis=0)
        print(f"{tag}: mean |SHAP| MEAN={mean_abs_shap[0]:.4f} RATIO={mean_abs_shap[1]:.4f}")

    print("Saved shap_summary_random_forest.png, shap_summary_xgboost.png")


# --------------------------------------------------------------- LSTM ---

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


def run_lstm_interpretability():
    with open(f"{P1R}/lstm_best_params.json") as f:
        params = json.load(f)

    dataset = pd.read_csv("Data/ATD_sequence.csv")
    X_all = dataset.drop("LABEL", axis=1).values.astype(np.float32)
    y_all = dataset["LABEL"].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)

    scaler = joblib.load(f"{P1M}/lstm_tuned_scaler.joblib")
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = LSTMClassifier(hidden_size=params["hidden_size"])
    model.load_state_dict(torch.load(f"{P1M}/lstm_tuned.pth", map_location="cpu"))
    model.eval()

    X_tensor = torch.tensor(X_test_s).unsqueeze(-1)
    n_timesteps = X_tensor.shape[1]

    with torch.no_grad():
        base_probs = torch.sigmoid(model(X_tensor)).numpy().flatten()
    base_preds = (base_probs > 0.5).astype(float)
    base_acc = (base_preds == y_test).mean()

    seq_mean = X_tensor.mean(dim=1, keepdim=True)  # (n, 1, 1)

    occlusion_importance = []
    permutation_importance = []

    rng = np.random.RandomState(SEED)

    for t in range(n_timesteps):
        # occlusion: replace timestep t with the sequence's own mean
        X_occ = X_tensor.clone()
        X_occ[:, t, :] = seq_mean.squeeze(1)
        with torch.no_grad():
            probs_occ = torch.sigmoid(model(X_occ)).numpy().flatten()
        occlusion_importance.append(float(np.mean(np.abs(base_probs - probs_occ))))

        # permutation: shuffle timestep t's values across the batch
        X_perm = X_tensor.clone()
        perm_idx = rng.permutation(X_perm.shape[0])
        X_perm[:, t, :] = X_tensor[perm_idx, t, :]
        with torch.no_grad():
            probs_perm = torch.sigmoid(model(X_perm)).numpy().flatten()
        preds_perm = (probs_perm > 0.5).astype(float)
        permutation_importance.append(float(base_acc - (preds_perm == y_test).mean()))

    # Integrated Gradients (captum) against a zero baseline
    def forward_fn(x):
        return torch.sigmoid(model(x))

    ig = IntegratedGradients(forward_fn)
    baseline = torch.zeros_like(X_tensor)
    # captum on the full 7,360-row test set at once can be heavy; use a stratified sample
    idx = rng.choice(len(X_tensor), size=min(1000, len(X_tensor)), replace=False)
    attributions = ig.attribute(X_tensor[idx], baseline[idx], n_steps=50)
    ig_importance = attributions.abs().mean(dim=0).squeeze(-1).detach().numpy()  # (n_timesteps,)

    df = pd.DataFrame({
        "timestep": list(range(1, n_timesteps + 1)),
        "occlusion_importance": occlusion_importance,
        "permutation_importance": permutation_importance,
        "integrated_gradients_importance": ig_importance.tolist(),
    })
    df.to_csv(f"{OUT}/results/lstm_timestep_importance.csv", index=False)

    plt.figure(figsize=(11, 6))
    plt.plot(df["timestep"], df["occlusion_importance"] / df["occlusion_importance"].max(),
              marker="o", label="Occlusion (normalized)")
    plt.plot(df["timestep"], df["permutation_importance"] / df["permutation_importance"].max(),
              marker="s", label="Permutation (normalized)")
    plt.plot(df["timestep"], df["integrated_gradients_importance"] / df["integrated_gradients_importance"].max(),
              marker="^", label="Integrated Gradients (normalized)")
    plt.xlabel("Timestep")
    plt.ylabel("Relative importance")
    plt.title("LSTM: which timesteps drive the decision?")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/lstm_timestep_importance.png", dpi=300)
    plt.close()

    print("LSTM timestep importance (top 5 by Integrated Gradients):")
    print(df.sort_values("integrated_gradients_importance", ascending=False).head(5).to_string(index=False))
    print("Saved lstm_timestep_importance.csv, lstm_timestep_importance.png")


# -------------------------------------------------------- Autoencoder ---

class Autoencoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(50, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 50))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder_interpretability():
    with open(f"{P1R}/autoencoder_best_params.json") as f:
        params = json.load(f)

    dataset = pd.read_csv("Data/ATD_sequence.csv")
    X_all = dataset.drop("LABEL", axis=1).values.astype(np.float32)
    y_all = dataset["LABEL"].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)

    scaler = joblib.load(f"{P1M}/autoencoder_tuned_scaler.joblib")
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = Autoencoder(latent=params["latent"])
    model.load_state_dict(torch.load(f"{P1M}/autoencoder_tuned.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        recon = model(torch.tensor(X_test_s)).numpy()

    per_timestep_error = (X_test_s - recon) ** 2  # (n, 50)

    normal_err = per_timestep_error[y_test == 0].mean(axis=0)
    attack_err = per_timestep_error[y_test == 1].mean(axis=0)

    plt.figure(figsize=(11, 6))
    plt.plot(range(1, 51), normal_err, marker="o", label="Normal (mean per-timestep error)")
    plt.plot(range(1, 51), attack_err, marker="s", label="Attack (mean per-timestep error)")
    plt.xlabel("Timestep")
    plt.ylabel("Mean squared reconstruction error")
    plt.title("Autoencoder: which timesteps reconstruct worst under attack?")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/autoencoder_timestep_reconstruction_error.png", dpi=300)
    plt.close()

    pd.DataFrame({"timestep": range(1, 51), "normal_mean_error": normal_err, "attack_mean_error": attack_err}).to_csv(
        f"{OUT}/results/autoencoder_timestep_reconstruction_error.csv", index=False
    )
    print(f"Autoencoder: worst-reconstructed timestep under attack = "
          f"{int(np.argmax(attack_err)) + 1} (error={attack_err.max():.4f})")
    print("Saved autoencoder_timestep_reconstruction_error.csv/.png")


if __name__ == "__main__":
    print("=== SHAP: Random Forest / XGBoost ===")
    run_tree_shap()

    print("\n=== LSTM: occlusion / permutation / Integrated Gradients ===")
    run_lstm_interpretability()

    print("\n=== Autoencoder: per-timestep reconstruction error ===")
    run_autoencoder_interpretability()
