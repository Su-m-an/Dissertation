# Methodology

Working draft for the dissertation's methodology chapter. Written from
the code and the recovered MATLAB source, meant as a starting point to
edit into your own voice rather than final copy.

## 1. System model

The scenario is active eavesdropping through pilot contamination. A base
station with `L = 10` receive antennas serves `K = 4` legitimate users,
each assigned an orthogonal pilot sequence for channel estimation. An
eavesdropper attempts to impersonate a target user by transmitting on
that user's pilot, so its signal cannot be separated from the legitimate
one by pilot orthogonality alone.

For a target user `k`, over `T = 50` time samples, the received signal in
the non-attack case is

```
y_non(t) = sqrt(L * rho_u) * p_k * g_k(t) + n(t)
```

and, when an eavesdropper is present,

```
y_attack(t) = sqrt(L * rho_u) * p_k * g_k(t) + sqrt(L * rho_E) * p_k * g_E(t) + n(t)
```

where `g_k(t)` and `g_E(t)` are independent Rayleigh-fading channel
gains for the legitimate user and the eavesdropper (`beta_k = beta_E = 1`),
`n(t)` is complex Gaussian receiver noise, and `rho_u`, `rho_E` are the
legitimate user's and eavesdropper's SNR. The base station applies a
matched filter for the target pilot, `p_k`, and records the received
power

```
z(t) = | p_k^H * y(t) |^2
```

Detection reduces to distinguishing the statistics of `z_non(t)` from
`z_attack(t)` over the observation window. `rho_E` is the parameter that
determines how hard this is: at low `rho_E` the eavesdropper's
contribution is buried in noise: near `rho_E = rho_u`, its contribution
is comparable to the legitimate signal, and easy to spot.

Baseline parameters, matching the existing datasets: `K=4, L=10, T=50,
rho_u=rho_E=5, beta_k=beta_E=1`. See
`experiments/phase3_physical_layer/` for what happens as `rho_E` moves
away from that point.

## 2. Datasets

Two representations of the same underlying signal are used.

**Feature-based** (`Data/ATD.csv`, `Data/ATD_features.csv`): for each of
`T_hat = 2,000` Monte Carlo trials and each window length
`tt = T0..T` (`T0 = 5`), two summary statistics are computed over
`z(1:tt)`:

- `MEAN`: the mean received power over the window
- `RATIO`: excess power relative to the noise floor,
  `(sum(z) - sum(noise_power)) / sum(noise_power)`

This produces `T_hat * 2 * (T - T0 + 1) = 184,000` rows, half attack and
half non-attack.

**Sequence-based** (`Data/ATD_sequence.csv`): the raw 50-step power trace
`z(1:T)` for each of `T_hat * 2 = 4,000` trials, unaggregated.

Both `ATD.csv` and `ATD_features.csv` are independent draws from the same
generator at identical parameters (the MATLAB source sets no explicit
random seed, so every run produces a fresh Monte Carlo sample); this is
what makes `ATD_features.csv` usable as an external validation set for
models trained on `ATD.csv` (`src/14_external_validation.py`, extended to
five replicates in `experiments/phase3_physical_layer/`).

## 3. Models

| Model | Input | Notes |
|---|---|---|
| TC-SVM | MEAN, RATIO | RBF kernel, following Hoang et al.'s reported parameters as a starting point, retuned in Phase 1 |
| Random Forest | MEAN, RATIO | Ensemble of decision trees |
| XGBoost | MEAN, RATIO | Gradient-boosted trees |
| MLP | E1..E50 (flattened) | Fully connected, no temporal structure |
| Autoencoder | E1..E50 | Trained on normal sequences only; flags a sequence as an attack when its reconstruction error exceeds a threshold calibrated on held-out normal data |
| LSTM | E1..E50 (sequential) | Two-layer recurrent network, processes the window as a time series |

All models are evaluated with an 80/20 stratified train/test split
(`random_state=42`). Feature scaling (`StandardScaler`, fit on training
data only) is applied throughout; this was found and fixed as a real bug
for the sequence models early in this project (see the git history for
`src/11_mlp.py`, `src/12_autoencoder.py`, `src/13_lstm_final.py`), where
its absence had significantly understated MLP and Autoencoder
performance.

## 4. Evaluation protocol

The baseline comparison (`src/15_compare_models.py`, `results/`) reports
a single train/test split per model. `experiments/` extends this in
three phases.

### Phase 1: statistical rigor

- **Cross-validation**: stratified 5-fold CV for every model, reported as
  mean ± SD rather than a point estimate. TC-SVM's hyperparameters are
  searched on a stratified subsample (RBF-kernel training cost scales
  roughly `O(n^2.2` to `n^2.5)`, making a full grid at 184,000 rows
  impractical); the winning configuration is then evaluated with real
  5-fold CV at full scale.
- **Significance testing**: Wilcoxon signed-rank across CV folds (chosen
  over a paired t-test, since the normality assumption a t-test needs is
  hard to justify with only 5 folds) and McNemar's test on paired
  test-set predictions, both valid only *within* the classical
  {TC-SVM, RF, XGBoost} and neural {MLP, LSTM, Autoencoder} groups, since
  each group shares identical folds and test rows while the two groups
  do not (different underlying datasets). Cross-group comparisons use
  the weaker, unpaired Mann-Whitney U test, reported as such.
- **Calibration and cost**: reliability diagrams, PR-AUC, inference
  latency, and model size/parameter count for every model.

### Phase 2: depth and robustness

- **Class imbalance**: all six models re-evaluated at 90/10, 95/5, 99/1,
  and 99.9/0.1 attack ratios (the last skipped for the smaller sequence
  dataset where too few positive samples would remain). Precision,
  recall, F1, PR-AUC, FPR, and FNR are the headline metrics here, not
  accuracy, which is misleading under severe imbalance.
- **Error analysis**: characterizes where each tuned model is wrong,
  using its saved test-set predictions.
- **Representation and sequence-length ablation**: isolates whether the
  LSTM's advantage over MLP comes from its architecture or from having
  more information, by holding model capacity fixed and varying only the
  input representation (from a single MEAN statistic up through the full
  50-step sequence), and separately comparing MLP against LSTM at each
  sequence length.
- **Interpretability**: SHAP for the tree models; timestep occlusion,
  permutation, and Integrated Gradients for the LSTM (which has no
  reconstruction target, so autoencoder-style language does not apply to
  it); per-timestep reconstruction error for the autoencoder.
- **Evasion robustness**: black-box random-direction search for the
  smallest feature-space perturbation that flips a correctly-detected
  attack sample's prediction. Explicitly framed as feature-space
  robustness, not a claim about physically achievable attacker actions,
  since grounding it in the physical simulator was not yet possible when
  this was run.

### Phase 3: physical-layer depth

- **SNR/difficulty sweep**: `rho_E` swept across
  `{0.1, 0.5, 1, 2, 5, 10, 20}` with all other parameters fixed, showing
  how detection accuracy changes as the eavesdropper's signal strength
  moves from far below to far above the legitimate user's. Model
  architecture/hyperparameters are fixed to the Phase 1 tuned values
  throughout, so the sweep isolates physical difficulty as the only
  variable, at the cost of occasionally showing what happens when a
  fixed architecture is pushed outside the regime it was tuned for (see
  the MLP overfitting result at the lowest SNR point, diagnosed directly
  via train/test AUC divergence rather than assumed).
- **External validation replicates**: extended from one independent
  simulation run to five, reporting generalization as a distribution
  rather than a single point estimate.
- **Regularization ablation**: follow-up on the MLP overfitting finding
  above, testing whether dropout, weight decay, and a smaller architecture
  (individually and combined) fix it at the two lowest SNR points, each
  retrained across 5 seeds. Regularization substantially reduces the
  train/test AUC gap and, combined with a smaller architecture, restores
  above-chance test AUC at `rho_E=0.5`, but does not fully recover it at
  `rho_E=0.1`, the hardest point in the sweep. Architecture size on its
  own, without dropout, barely helps: dropout is doing most of the work.

## 5. Reproducibility

- Every script fixes `random_state=42` (or an explicit seed passed to
  MATLAB's `rng()`) for its data splits and model initialization.
- The original MATLAB generators set no seed at all; every dataset
  produced without an explicit `rng()` call is therefore a fresh,
  unrepeatable Monte Carlo draw. `experiments/phase3_physical_layer/`
  adds explicit seeding on top of the unmodified original scripts.
- `requirements.txt` pins every dependency version used.
