# 4. Implementation

*Status: full draft. Describes the actual pipeline in this repository;*
*script names and paths are real, check them against the repo if this*
*changes structure before submission.*

## 4.1 Environment and Tools

The physical-layer simulation is generated in MATLAB (R2025a); everything
downstream is Python. A dedicated virtual environment
(`dissertation-venv`) pins every dependency in `requirements.txt`,
notably `scikit-learn` (classical models, cross-validation, metrics),
`xgboost`, `torch` (MLP, LSTM, Autoencoder), `shap` and `captum`
(interpretability, Chapter 6), `pandas`/`numpy` (data handling), and
`joblib` (model/scaler persistence). Reproducing any result in this
dissertation only requires `pip install -r requirements.txt` into this
environment and running the relevant script; no result depends on
unpinned or system-level package versions.

Every package version is pinned rather than left as a floating
`>=` requirement, a deliberate choice: floating versions are the more
common default, but a floating `scikit-learn` or `torch` version means a
tuned hyperparameter (§ 3.4) or a random seed (§ 3.5) can silently stop
reproducing the same result months later when the library's internals
change. For a dissertation whose evaluation protocol leans heavily on
exact reproducibility (cross-validation folds, fixed seeds, calibrated
thresholds), pinning is treated as a correctness requirement, not just
good practice.

The development and execution environment is headless (no attached
display), which mattered directly during implementation: see § 4.3 for
the `plt.show()` fix this required.

## 4.2 Pipeline Overview

The pipeline runs in two stages, matching the data representations
described in Chapter 3.

```
Script Dataset/*.m  -->  Data/*.csv  -->  src/01..15 (baseline)  -->  results/
                                       -->  experiments/phase1..3 -->  experiments/*/results/
```

![Figure 4.1: Pipeline architecture, from the MATLAB simulator through the two data representations to the six models and the evaluation layer.](images/fig4_1_pipeline_architecture.png)

**Figure 4.1.** The two representations of `z(t)` (§ 3.2) fan out to two
disjoint groups of models; every model's output converges on a shared
evaluation layer, which is what makes the like-for-like comparison in
Chapter 5 possible despite the two families never touching the same
underlying rows.

The numbered `src/` scripts form the baseline pipeline, run in order:

| Stage | Scripts | Produces |
|---|---|---|
| Data loading, visualisation, preprocessing | `01`-`03` | EDA plots, cleaned feature frame |
| TC-SVM | `04`-`06` | Trained model, ROC curve, metrics |
| Random Forest | `07` | Trained model, feature importance, metrics |
| XGBoost | `08` | Trained model, feature importance, metrics |
| Classical model comparison | `09` | `results/model_comparison.csv` |
| Sequence loading | `10` | Scaled sequence tensors |
| MLP | `11` | Trained model, metrics |
| Autoencoder | `12` | Trained model, calibrated threshold, metrics |
| LSTM | `13` (`13_lstm_final.py`; `13_lstm.py` kept for reference, superseded) | Trained model, metrics |
| External validation | `14` | Classical models scored on an independent simulation run |
| Final comparison | `15` | `results/final_model_comparison.csv` |

`experiments/phase1_statistical_rigor/`, `phase2_depth_robustness/`, and
`phase3_physical_layer/` each write only to their own `results/`
subdirectory; none of them modify `results/` at the repository root, so
the baseline numbers in Chapter 5's headline table and the deeper
experiment results can always be traced back to the exact script that
produced them.

## 4.3 Data Loading and Exploratory Analysis (`src/01`-`03`)

`src/01_data_loading.py` loads `Data/ATD.csv`, `Data/ATD_features.csv`,
and `Data/ATD_sequence.csv` with `pandas.read_csv`, and checks class
balance, missing values (none found; the MATLAB generator produces
complete rows by construction), and the expected row counts derived in
Chapter 3 (184,000 rows for the feature-based sets, split evenly 50/50
between the `LABEL=0` and `LABEL=1` classes; 4,000 for the sequence set,
also balanced). This balance is a property of the generator, not an
accident: `ATD_generator.m` emits exactly one non-attack and one attack
row per `(trial, window length)` combination (§ 3.2), so class imbalance
is never a concern for the baseline pipeline, only for the deliberately
imbalanced re-sampling done later in § Class imbalance stress test
(Chapter 5/6).

`src/02_data_visualization.py` plots the two classical features split by
class, the natural first sanity check before training anything: if the
classes were not visibly separable here, no downstream classifier result
would be trustworthy.

![Figure 4.2: Distribution of the MEAN feature, split by class.](images/fig4_2_mean_distribution.png)

**Figure 4.2.** `MEAN` distribution by class. The attack-class
distribution is visibly shifted to higher power, consistent with the
worked example in § 3.2.1: the eavesdropper's contribution adds
directly to the received power at every timestep, so its mean is higher
whenever `rho_E > 0`.

![Figure 4.3: Distribution of the RATIO feature, split by class.](images/fig4_3_ratio_distribution.png)

**Figure 4.3.** `RATIO` distribution by class, showing a comparable, and
visually sharper, separation to `MEAN`. Both distributions overlap at
their tails, which is exactly the region every classifier's errors in §
5.9/6.9 are drawn from.

![Figure 4.4: Scatter of MEAN against RATIO, coloured by class.](images/fig4_4_scatter_mean_ratio.png)

**Figure 4.4.** `MEAN` vs. `RATIO`, coloured by class. The two features
are strongly correlated (both are monotonic functions of the same
underlying received power), which is itself informative: it means the
two-feature classical representation carries less independent
information than two arbitrary features would, a fact that connects
directly to § 6.2's finding that a single feature (`MEAN` alone) captures
almost everything `MEAN+RATIO` together does.

One implementation issue found and fixed here: the original visualisation
and evaluation scripts (`02`, `05`-`08`) called `plt.show()`, which blocks
indefinitely in a headless execution environment, exactly the environment
this pipeline runs in (§ 4.1). Running any of these scripts unmodified
would hang indefinitely at the first plot rather than completing; this
was caught by direct observation (a script simply never returning) rather
than by a test, since a hang produces no error message to catch. All
affected scripts are changed to `plt.close()` immediately after saving
each figure to disk, so the full pipeline can run unattended end-to-end,
including in the batch/background execution this dissertation's
longer-running experiments (§ 4.6) depend on.

## 4.4 Classical Models (`src/04`-`09`)

TC-SVM, Random Forest, and XGBoost are each trained on `[MEAN, RATIO]`
via an 80/20 stratified `train_test_split(random_state=42)`
(`sklearn.model_selection`), using scikit-learn/XGBoost defaults as a
starting point (`src/04_tc_svm.py`, `src/07_random_forest.py`,
`src/08_xgboost.py`), then tuned via the documented search described in
Chapter 3 § 3.4. Stratification here matters more than it might for a
generic classification problem: because the feature set already spans
every window length `T0`-`T` (§ 3.2), an unstratified split could,
by chance, concentrate short-window (harder) samples in the test set and
long-window (easier) samples in training, distorting the reported metric
independent of any real model quality difference.

Concretely, the classical training step follows the same three-line
pattern in each of `04`, `07`, and `08`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

with `model` instantiated as `SVC(kernel="rbf", C=..., gamma=...)`,
`RandomForestClassifier(max_depth=..., n_estimators=..., random_state=42)`,
or `XGBClassifier(learning_rate=..., max_depth=..., n_estimators=...,
random_state=42)` respectively. Unlike the sequence models (§ 4.5),
`[MEAN, RATIO]` is *not* scaled before TC-SVM/RF/XGBoost in the baseline
pipeline; this is a defensible choice for the tree-based models (split
decisions are scale-invariant) but is a real sensitivity for TC-SVM's RBF
kernel, whose `gamma` implicitly depends on feature scale. This is worth
flagging directly: the Phase 1 hyperparameter search (§ 3.4) retunes
`gamma` *for this specific unscaled feature range*, so the tuned value
`gamma=0.01` is not portable to a rescaled version of the same features
without re-tuning.

Each of `06`-`08` persists its trained model and (where applicable)
scaler to `saved_models/` via `joblib.dump()`, so that later stages
(`14_external_validation.py`, and every Phase 1-3 script that reuses a
tuned model) never need to retrain, only reload:

```python
joblib.dump(model, "saved_models/random_forest.joblib")
```

`05_roc_curve.py` and `06_model_evaluation.py` originally wrote an
identically-named ROC curve figure to the same path, so whichever script
ran second silently overwrote the other's output with no error or
warning, a class of bug that produces no incorrect *numbers* but can
silently produce a misleading *figure* (the visible ROC curve belonging
to a different run than the metrics text next to it). `05`'s output is
renamed to `roc_curve_tc_svm_standalone.png` to disambiguate; this class
of collision is checked for across the rest of the repository's output
paths as part of the broader code review (§ 4.7).

## 4.5 Deep Learning Models (`src/10`-`13`)

`src/10_sequence_loading.py` loads `Data/ATD_sequence.csv`, splits it
80/20 (stratified, `random_state=42`), and fits a `StandardScaler` on the
training split only, applying the same fitted transform to the test
split, before the MLP, Autoencoder, and LSTM are trained on the result.
All three are implemented in PyTorch with `torch.manual_seed(42)` and
`np.random.seed(42)` set before model construction and training, for
reproducibility. This scaling step is the fix described in Chapter 3 §
3.3: the original versions of `11`, `12`, and `13` trained directly on
raw, unscaled `E1..E50` values, and the accuracy jump this produced when
fixed (MLP 73% to 98.75%) is large enough that it is worth restating here
as an implementation lesson: for gradient-based models specifically,
missing or incorrect input scaling is not a minor tuning detail, it can
be the dominant source of reported performance difference between two
otherwise-identical experiments.

**MLP** (`src/11_mlp.py`). Instantiated directly from the architecture in
§ 3.3.4:

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(50, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)
```

trained for 30 epochs with `BCEWithLogitsLoss` and `Adam`. The baseline
script's learning rate (`0.001`) differs from the Phase 1 tuned value
(`0.002`, § 3.4); results reported from Chapter 5 onward use the tuned
value unless explicitly labelled "baseline."

![Figure 4.5: MLP training loss curve.](images/fig4_5_mlp_training_loss.png)

**Figure 4.5.** MLP training loss over 30 epochs, converging smoothly
with no visible instability, consistent with the model's small parameter
count (16,897) relative to the training set size at the baseline SNR.
Contrast this with the training behaviour underlying the low-SNR
overfitting finding in § 6.5, which shows up as a train/test *gap*, not
as instability in the training curve itself, a reminder that a smooth
training loss curve on its own says nothing about generalisation.

**LSTM** (`src/13_lstm_final.py`). A two-layer `nn.LSTM` with
`hidden_size=64` (baseline) and dropout `0.3` between layers, processing
the sequence one scalar timestep at a time (`input_size=1`), with the
final hidden state passed through a dropout layer and a linear
classification head:

```python
self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                     dropout=0.3, batch_first=True)
self.dropout = nn.Dropout(0.3)
self.fc = nn.Linear(64, 1)
```

trained for up to 50 epochs with early stopping (patience 5 epochs on a
held-out validation split), `BCEWithLogitsLoss`, and `Adam`. The Phase 1
tuned configuration (`hidden_size=32`, § 3.4) is smaller than this
baseline, one of the few cases in this dissertation where the documented
search selected a *smaller* network than the initial default, worth
noting since it is direct evidence the tuning process was not simply
selecting maximum capacity.

![Figure 4.6: LSTM training and validation loss curves.](images/fig4_6_lstm_training_loss.png)

**Figure 4.6.** LSTM training and validation loss. Both curves are shown
because the LSTM, unlike the MLP, uses early stopping against a
validation split (`patience=5`); the gap or convergence between the two
curves is what the stopping criterion is actually watching, unlike
Figure 4.5's single training curve.

**Autoencoder** (`src/12_autoencoder.py`). Encoder `50 -> 32 -> 16 -> 8`
(baseline latent dimension 8; Phase 1 tunes this to 4, § 3.4), mirrored
decoder, trained to minimise mean squared reconstruction error on
non-attack sequences only:

```python
X_train_normal = X_train_full[y_train_full == 0]
X_fit, X_threshold_calib = train_test_split(
    X_train_normal, test_size=0.20, random_state=42
)
# ... train model on X_fit only ...
calib_errors = np.mean((X_threshold_calib - calib_reconstructed) ** 2, axis=1)
threshold = np.percentile(calib_errors, 95)
```

This is the exact code implementing the threshold-calibration formula
given in § 3.3.6, and the exact split that fixes the leakage bug
described in § 3.3: `X_threshold_calib` is held out from `X_fit`, so the
95th-percentile threshold is computed from sequences the model never
trained on, and separately, both are drawn only from the training split,
never from `X_test`.

![Figure 4.7: Autoencoder training loss curve (reconstruction MSE on non-attack training sequences).](images/fig4_7_autoencoder_training_loss.png)

**Figure 4.7.** Autoencoder reconstruction-error training loss. Because
the autoencoder only ever sees non-attack sequences during training (§
3.3.6), this curve, unlike Figures 4.5/4.6, has no notion of
classification accuracy at all, it purely reflects how well the model
learns to compress and reconstruct normal traffic, the anomaly threshold
in § 4.6 is applied only after this training is complete.

## 4.6 Evaluation Methodology, Implemented

The baseline comparison (`src/15_compare_models.py`) collects each
model's held-out test-set predictions and computes accuracy, precision,
recall, F1, and AUC uniformly across all six models, regardless of family,
so that `results/final_model_comparison.csv` is a like-for-like table
(Chapter 5, Table 5.1).

The three experiment phases are implemented as independent script chains:

- **Phase 1** (`pilot_cv_check.py` first, validating the CV harness
  itself has no leakage and per-fold scalers are refit correctly, before
  `cv_tuning_classical.py`/`cv_tuning_neural.py`, `statistical_tests.py`,
  `calibration_and_latency.py` run the actual search, CV, and testing
  described in Chapter 3).
- **Phase 2** (`class_imbalance_stress_test.py`, `error_analysis.py`,
  `representation_and_sequence_ablation.py`, `interpretability.py`,
  `evasion_robustness.py`, each independent, reusing Phase 1's saved
  models/predictions where possible rather than retraining).
- **Phase 3** (`generate_sweep_data.sh` drives the unmodified MATLAB
  generator across the `rho_E` sweep and five external replicates, with
  explicit `rng()` seeding added; `snr_sweep_evaluation.py`,
  `mlp_low_snr_seed_check.py`, `external_validation_replicates.py`, and
  `mlp_regularization_ablation.py` consume the resulting data).

One implementation issue worth documenting here as a lesson rather than
hiding it: the first version of `snr_sweep_evaluation.py` held all results
in memory and wrote them to disk only once, at the very end, of what
turned out to be a roughly hour-long run. The process was killed partway
through by an unrelated tooling issue, and the entire run's results were
lost. The script was rewritten to append each result row to its output
CSV incrementally, with a resume/skip function that checks which
`(rho_E, model)` combinations are already recorded, so a future
interruption costs only the time since the last write rather than the
whole run. This is a reproducibility property worth stating explicitly
for any script in this repository that takes more than a few minutes to
run.

## 4.7 Code Review and Correctness Checks

Beyond the methodological corrections in Chapter 3 § 3.3, a code review
pass over `experiments/` found and fixed four further issues, listed here
for completeness since they affect confidence in the results rather than
their headline numbers: a missing `set -o pipefail` in
`generate_sweep_data.sh` that could let a MATLAB failure be silently
masked by a downstream `grep`; a missing `os.makedirs()` in
`mlp_low_snr_seed_check.py` that happened to work only because another
script had already created the same directory; `snr_sweep_evaluation.py`
reloading and rescaling the same large CSV three times per `rho_E` point
rather than once, refactored to load-once; and a missing
`zero_division=0` argument in `external_validation_replicates.py`'s
metric calls, for consistency with the rest of the codebase.
