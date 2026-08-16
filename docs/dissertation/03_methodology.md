# 3. Methodology

*Status: full draft. This describes work that was actually done and*
*verified, so it's written as prose you can edit for voice and emphasis,*
*not as a skeleton. Numbers, filenames, and script references are all*
*real and traceable into the repository; check them if you change*
*anything, don't just take the arithmetic on trust.*

## 3.1 System Model

This work adopts the pilot-contamination attack model of Hoang et al.
(2021) [1]. A base station equipped with `L = 10` receive antennas serves
`K = 4` legitimate users in the uplink of a single cell, each assigned an
orthogonal pilot sequence `p_k` for channel estimation. An active
eavesdropper attacks this process by transmitting on a target user `k`'s
pilot during the training phase, so that its signal cannot be separated
from the legitimate user's by pilot orthogonality alone, a pilot
contamination attack.

For the target user, over an observation window of `T = 50` time samples,
the received signal at the base station is

```
y_non(t)    = sqrt(L * rho_u) * p_k * g_k(t) + n(t)                                 (no attack)
y_attack(t) = sqrt(L * rho_u) * p_k * g_k(t) + sqrt(L * rho_E) * p_k * g_E(t) + n(t) (attack)
```

where `g_k(t)` and `g_E(t)` are independent Rayleigh-fading channel gains
for the legitimate user and the eavesdropper (fading variance
`beta_k = beta_E = 1`), `n(t)` is complex Gaussian receiver noise, and
`rho_u`, `rho_E` are the legitimate user's and eavesdropper's transmit
SNR respectively. The base station applies a matched filter for the
target pilot and records the received power at each time step,

```
z(t) = | p_k^H * y(t) |^2
```

Detection is therefore the problem of distinguishing the statistics of
`z_non(t)` from `z_attack(t)` over the observation window, using only
`z(t)`, without direct access to `g_k`, `g_E`, or `n`. The parameter
`rho_E` controls how hard this is: at low `rho_E` the eavesdropper's
contribution to `z(t)` is buried in the noise floor and effectively
indistinguishable; as `rho_E` approaches `rho_u`, its contribution becomes
comparable to the legitimate signal and easier to detect. The baseline
configuration used throughout this dissertation, unless stated otherwise,
is `K=4, L=10, T=50, rho_u=rho_E=5, beta_k=beta_E=1`; Chapter 5/6 report
what happens as `rho_E` is varied independently of `rho_u` (§ Physical
layer sweep).

### 3.1.1 Detection as a Binary Hypothesis Test

Written out formally, the base station observes `z(t)` for `t = 1..tt`
and must decide between two hypotheses:

```
H0 (no attack): z(t) = | p_k^H ( sqrt(L*rho_u) * p_k * g_k(t) + n(t) ) |^2
H1 (attack):     z(t) = | p_k^H ( sqrt(L*rho_u) * p_k * g_k(t) + sqrt(L*rho_E) * p_k * g_E(t) + n(t) ) |^2
```

With `p_k` a unit-norm pilot (`p_k^H p_k = 1`, as implemented: `p_k` is a
standard basis vector in the code) and `n(t)` circularly symmetric complex
Gaussian noise with unit variance per antenna, `p_k^H n(t)` is itself a
standard complex Gaussian, so the noise-only component of the received
power,

```
noise_power(t) = | p_k^H n(t) |^2
```

is exponentially distributed. Under `H0`, `z(t)` is a scaled non-central
chi-squared statistic driven by `g_k(t)`; under `H1`, it additionally
carries the eavesdropper's independent contribution scaled by `rho_E`.
An optimal (Neyman-Pearson) detector would form the likelihood ratio
between these two distributions, but doing so in closed form requires
knowing `rho_E` and `beta_E`, the eavesdropper's transmit power and
fading statistics, neither of which is available to the legitimate
receiver in practice, that's precisely the setting an eavesdropper would
choose not to reveal. This is the motivation, shared with Hoang et al.
[1], for a data-driven detector: rather than deriving an analytical test
statistic that depends on unknown attacker parameters, a classifier is
trained on labelled examples of `z(t)` under both hypotheses (generated
here via Monte Carlo simulation, § 3.2) and learns a decision boundary
directly.

![Figure 3.1: System model. A base station with L=10 antennas serves K legitimate users over orthogonal pilots; an eavesdropper attacks by transmitting on the target user's pilot p_k during the training phase.](images/fig3_1_system_model.png)

**Figure 3.1.** System model schematic. The base station cannot distinguish
the eavesdropper's contamination from the legitimate signal by pilot
orthogonality alone, since the eavesdropper deliberately reuses `p_k`;
all it can observe is the aggregate received power `z(t)`.

This system model, and the two datasets described next, were recovered
independently from the project's MATLAB source (`Script Dataset/`) before
being cross-checked against Hoang et al.'s published description; the two
match, which is itself a form of validation of both the recovered model
and the implementation.

## 3.2 Datasets

Two representations of the same underlying signal, `z(t)`, are used by
different models in this dissertation.

**Feature-based representation** (`Data/ATD.csv`, `Data/ATD_features.csv`).
For each of `T_hat = 2,000` Monte Carlo trials, and for every window
length `tt` from `T0 = 5` up to `T = 50`, two summary statistics are
computed over the partial trace `z(1:tt)`:

- `MEAN`: the mean received power over the window.
- `RATIO`: excess power relative to the expected noise floor,
  `(sum(z) - sum(noise_power)) / sum(noise_power)`.

This produces `T_hat * 2 * (T - T0 + 1) = 184,000` rows, balanced between
attack and non-attack classes. This is the representation used by the
three classical models (TC-SVM, Random Forest, XGBoost), matching Hoang
et al.'s feature construction.

**Sequence representation** (`Data/ATD_sequence.csv`). The raw 50-step
power trace `z(1:T)` for each of `T_hat * 2 = 4,000` trials,
unaggregated, columns `E1` through `E50`. This is the representation used
by the three deep learning models (MLP, LSTM, Autoencoder), and is the
representation this dissertation's central comparison is about: does
giving a model the full trace, rather than the two-statistic summary
above, improve detection?

Both `ATD.csv` and `ATD_features.csv` are independent draws from the same
generator at identical parameters. The original MATLAB scripts set no
explicit random seed, so every execution produces a fresh, statistically
independent Monte Carlo sample. This is a useful property that this
dissertation exploits directly: `ATD_features.csv`, being an independent
draw from the same distribution as `ATD.csv`, serves as an external
validation set for models trained on `ATD.csv` (§ 3.4, and
`src/14_external_validation.py`), later extended to five independent
replicates (§ Physical layer robustness, `experiments/phase3_physical_layer/`).

### 3.2.1 Worked Example

To make the abstract feature definitions concrete: consider a single
Monte Carlo trial at the baseline operating point
(`rho_u = rho_E = 5, beta_k = beta_E = 1`) and a short window `tt = 5`.
The base station records five received-power samples
`z(1), ..., z(5)` and, separately (available only to the simulator, not
the detector), the noise-only power `noise_power(1), ..., noise_power(5)`
that would have been observed with no user signal at all. Suppose, for
this trial,

```
z_non    = [4.8, 5.9, 3.1, 6.4, 5.0]      (no attack)
z_attack = [11.2, 13.6, 9.8, 15.1, 12.4]  (attack present)
noise_power = [1.0, 0.8, 1.2, 0.9, 1.1]
```

Then

```
MEAN_non    = mean(z_non)    = 5.04
MEAN_attack = mean(z_attack) = 12.42

RATIO_non    = (sum(z_non) - sum(noise_power)) / sum(noise_power)    = (25.2 - 5.0) / 5.0    = 4.04
RATIO_attack = (sum(z_attack) - sum(noise_power)) / sum(noise_power) = (62.1 - 5.0) / 5.0    = 11.42
```

Both `MEAN` and `RATIO` roughly double under attack in this illustrative
example, since the eavesdropper's contribution (`rho_E = rho_u` here)
adds a second, comparably-sized power term to every sample. This is the
separation the classical models learn to detect; § 5.6's ablation result,
that `MEAN` alone nearly matches the full sequence, is a direct
consequence of how cleanly this separation shows up in a single summary
statistic at this operating point. The sequence models, by contrast, see
the five (or fifty, at full window length) individual values
`z(1), ..., z(tt)` rather than their mean, and must learn to extract an
equivalent, or better, decision statistic themselves.

## 3.3 Models

Six detectors are implemented and compared, split into two families by
input representation.

| Model | Input | Notes |
|---|---|---|
| TC-SVM | `MEAN, RATIO` | RBF kernel; parameters retuned via documented search (§ 3.4), starting from Hoang et al.'s reported configuration |
| Random Forest | `MEAN, RATIO` | Ensemble of decision trees |
| XGBoost | `MEAN, RATIO` | Gradient-boosted trees |
| MLP | `E1..E50` (flattened) | Fully connected network; sees the full sequence but with no explicit temporal structure |
| LSTM | `E1..E50` (sequential) | Two-layer recurrent network; processes the window as an ordered time series |
| Autoencoder | `E1..E50` | Trained only on non-attack sequences; flags a test sequence as an attack when its reconstruction error exceeds a threshold calibrated on held-out normal data |

### 3.3.1 TC-SVM

The classical baseline follows Hoang et al.'s Time-Correlation SVM
(TC-SVM): a soft-margin support vector classifier over the two-dimensional
feature vector `x = [MEAN, RATIO]`. Training solves the standard dual
soft-margin problem

```
maximise_alpha   sum_i(alpha_i) - (1/2) * sum_i sum_j( alpha_i * alpha_j * y_i * y_j * K(x_i, x_j) )
subject to       0 <= alpha_i <= C,   sum_i( alpha_i * y_i ) = 0
```

with the RBF kernel

```
K(x_i, x_j) = exp( -gamma * || x_i - x_j ||^2 )
```

and the decision function

```
f(x) = sign( sum_i( alpha_i * y_i * K(x_i, x) ) + b )
```

The baseline pipeline (`src/04_tc_svm.py`) uses `C=1, gamma=0.001` as a
starting point; the documented search in § 3.4 retunes this to
`C=10, gamma=0.01`, used for every TC-SVM result reported from Chapter 5
onward unless stated otherwise.

### 3.3.2 Random Forest and 3.3.3 XGBoost

Both are tree-ensemble classifiers over the same `[MEAN, RATIO]` feature
vector. Random Forest averages the predictions of `n_estimators`
decision trees, each trained on a bootstrap resample of the training data
with a random subset of features considered at each split (bagging),
splitting nodes to minimise Gini impurity. XGBoost instead builds trees
sequentially, each new tree `f_m` fit to the negative gradient of a
regularised loss with respect to the current ensemble's predictions,

```
L(phi) = sum_i( l(y_i, y_hat_i) ) + sum_m( Omega(f_m) ),      Omega(f) = gamma*T + (1/2)*lambda*||w||^2
```

where `l` is the binary logistic loss, `T` is the number of leaves in
tree `f_m`, and `w` its leaf weights, `gamma` and `lambda` penalise tree
complexity. Tuned hyperparameters for both (§ 3.4, Chapter 5, Table 5.2)
are `{max_depth: 10, n_estimators: 300}` for Random Forest and
`{learning_rate: 0.1, max_depth: 4, n_estimators: 100}` for XGBoost.

### 3.3.4 MLP

The MLP classifies the full flattened sequence `[E1, ..., E50]` with
three hidden layers of width 128, 64, 32 (ReLU activation), and a single
linear output unit:

```
h1 = ReLU(W1 x + b1),   h2 = ReLU(W2 h1 + b2),   h3 = ReLU(W3 h2 + b3),   logit = W4 h3 + b4
```

trained with the binary cross-entropy loss applied to the sigmoid of the
logit (`BCEWithLogitsLoss`, numerically stable combination of the two),

```
L = -[ y * log(sigmoid(logit)) + (1-y) * log(1 - sigmoid(logit)) ]
```

and the Adam optimiser. The baseline architecture (`src/11_mlp.py`) uses
`lr=0.001`; the tuned configuration from § 3.4 (`hidden: [128,64,32],
lr: 0.002`) is used for Chapter 5 onward. Note that the hidden-layer
widths did not change under tuning, only the learning rate did, capacity
was held fixed while other hyperparameters were searched.

### 3.3.5 LSTM

The LSTM processes `[E1, ..., E50]` as an ordered sequence, one scalar
timestep at a time, through a two-layer recurrent network. Each LSTM
cell updates a hidden state `h_t` and cell state `c_t` via input, forget,
and output gates:

```
i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)      (input gate)
f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)      (forget gate)
o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)      (output gate)
c_hat_t = tanh(W_c [h_{t-1}, x_t] + b_c)
c_t = f_t * c_{t-1} + i_t * c_hat_t
h_t = o_t * tanh(c_t)
```

The final hidden state `h_T` (after processing the full 50-step window)
is passed through a dropout layer and a single linear classification
head, trained with the same `BCEWithLogitsLoss` as the MLP. The tuned
configuration (§ 3.4) uses `hidden_size=32, lr=0.0005`; the baseline
script (`src/13_lstm_final.py`) uses `hidden_size=64` with dropout `0.3`
between layers, early-stopped on a validation split (patience 5 epochs).

### 3.3.6 Autoencoder

The autoencoder is methodologically distinct from the other five: it is a
single-class (anomaly detection) model, trained without ever seeing an
attack example, whereas the other five are supervised binary classifiers.
This is a deliberate inclusion, not an oversight, since it mirrors Hoang
et al.'s own SC-SVM variant and answers a genuinely different deployment
question: can an attack be detected without ever having labelled attack
data to train on? § 3.4 and Chapter 6 return to this distinction.

Architecturally, an encoder compresses the 50-step sequence to a latent
vector `z = Enc(x)` (baseline: `50 -> 32 -> 16 -> 8`, ReLU activations;
tuned configuration, § 3.4: latent dimension `4`), and a decoder
reconstructs `x_hat = Dec(z)` through the mirrored architecture, trained
to minimise mean squared reconstruction error on non-attack sequences
only:

```
L = (1/N) * sum_j( (x_j - x_hat_j)^2 )
```

At inference, a test sequence's reconstruction error
`e(x) = (1/50) * sum_j( (x_j - x_hat_j)^2 )` is compared against a fixed
threshold, calibrated (`src/12_autoencoder.py`) as the 95th percentile of
reconstruction error on a held-out slice of *normal* training data, never
on the test set:

```
threshold = percentile_95( { e(x) : x in X_calibration, X_calibration subset of normal training data } )
decision(x) = attack  if e(x) > threshold,  else non-attack
```

This threshold-calibration detail matters beyond implementation trivia:
it is the exact mechanism by which the leakage bug described below was
introduced, and the exact mechanism of its fix.

All models are evaluated with a stratified 80/20 train/test split
(`random_state=42`) as a baseline, extended to 5-fold cross-validation for
the primary results reported in Chapter 5 (§ 3.4). Feature scaling
(`StandardScaler`, fit on training data only, never on the test fold) is
applied throughout for the sequence models. This was found, during initial
development, to be a real and consequential bug: the sequence models were
originally trained on unscaled input, which significantly understated
MLP and autoencoder performance (MLP accuracy rose from 73% to 98.75% once
fixed). This is noted here because it is methodologically relevant, not as
a narrative aside: correct scaling is a precondition for the model
comparison in Chapter 5 to be meaningful at all, and it is worth stating
plainly that it was not correct on the first attempt.

A second, more serious methodological error was found and fixed in the
autoencoder specifically: the original implementation both trained on
sequences that were also used for reconstruction-error evaluation, and
calibrated its anomaly threshold on the (attack-contaminated) test set
rather than on held-out normal data alone, a form of train-on-test and
threshold leakage. This was corrected to a proper split, threshold
calibrated only on a held-out slice of normal training data. The effect
was large: recall rose from 0.10 to 0.99 and accuracy from 0.55 to 0.97
after the fix. This is reported here, rather than only in the git history,
because it is the single most consequential methodological correction in
this dissertation, and because a reader evaluating the autoencoder's
results in Chapter 5 should understand the result reflects a corrected,
leakage-free evaluation, not the original one.

## 3.4 Evaluation Protocol

### 3.4.1 Metrics

With `TP`, `TN`, `FP`, `FN` the counts of true/false positive/negative
predictions (positive = attack present), every result table in Chapter 5
uses:

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)                         (also: True Positive Rate, sensitivity)
F1        = 2 * Precision * Recall / (Precision + Recall)
FPR       = FP / (FP + TN)
FNR       = FN / (FN + TP) = 1 - Recall
```

AUC is the area under the Receiver Operating Characteristic curve (True
Positive Rate vs. False Positive Rate as the decision threshold varies);
PR-AUC is the area under the Precision-Recall curve, preferred over AUC
as the primary metric under severe class imbalance (§ class imbalance,
below), since AUC can remain misleadingly high when the positive class is
rare while precision collapses.

### 3.4.2 Protocol Stages

The evaluation is organised in four stages of increasing depth, moving
from a single-split baseline to the statistically rigorous,
robustness-tested results reported in Chapters 5 and 6.

**Baseline (`src/`, `results/`).** A single stratified 80/20 train/test
split per model, reported as a point estimate. This establishes the
headline comparison but not its statistical reliability.

**Cross-validation and significance testing
(`experiments/phase1_statistical_rigor/`).** Every model is re-evaluated
with stratified 5-fold cross-validation, reported as mean ± standard
deviation rather than a single number. Hyperparameters for every model are
selected via a documented search (grids recorded in
`*_search_grid.json`); TC-SVM's search is run on a stratified 15,000-row
subsample with 3-fold CV, since RBF-kernel training cost scales
approximately `O(n^2.2` to `n^2.5)` and a full grid search at 184,000 rows
was not computationally practical, after which the winning configuration
is evaluated with real 5-fold CV at full scale. The other five models are
searched and evaluated at full scale throughout.

Differences between models are tested for statistical significance using
the Wilcoxon signed-rank test across the five paired CV fold scores
(chosen over a paired t-test, since the t-test's normality assumption is
difficult to justify with only five folds), and McNemar's test on paired
test-set predictions. Both are valid only *within* the two model families
that share identical folds and test rows: {TC-SVM, Random Forest,
XGBoost} on the feature-based dataset, and {MLP, LSTM, Autoencoder} on
the sequence dataset. Because the two families use different underlying
datasets (§ 3.2), *cross-family* comparisons (e.g. "is the LSTM
significantly better than TC-SVM?") cannot use a paired test; the weaker,
unpaired Mann-Whitney U test is used instead for these, and is reported
explicitly as such rather than implied to carry the same statistical
weight as the within-family tests.

Reliability diagrams, PR-AUC, inference latency (per-sample and
full-test-set), model size, and parameter count are also recorded for
every model at this stage, to support the deployment-cost discussion in
Chapter 6.

**Depth and robustness (`experiments/phase2_depth_robustness/`).** Four
further analyses, each using the Phase 1 tuned hyperparameters without
retraining unless stated:

- *Class imbalance*: all six models re-evaluated at attack:non-attack
  ratios of 90:10, 95:5, 99:1, and 99.9:0.1, with a minimum-15-positive-
  test-sample threshold below which a configuration is skipped rather
  than reported on too few examples to be meaningful. Precision, recall,
  F1, PR-AUC, false positive rate, and false negative rate are reported
  as the headline metrics here, not accuracy, which becomes misleading
  once the classes are this imbalanced.
- *Error analysis*: characterises where each tuned model is wrong, using
  its saved test-set predictions, without retraining.
- *Representation and sequence-length ablation*: isolates whether the
  LSTM's advantage over the MLP (if any) comes from its recurrent
  architecture or simply from having access to more information, by
  holding model capacity fixed and varying only the input representation,
  from a single `MEAN` statistic, up through `E1`-`E5`, `E1`-`E10`, ...,
  `E1`-`E50`, and separately comparing MLP against LSTM at each sequence
  length under this controlled design.
- *Interpretability*: SHAP (TreeExplainer) for the tree-based models;
  timestep occlusion, permutation importance, and Integrated Gradients
  (via `captum`) for the LSTM, which has no reconstruction target and so
  cannot use autoencoder-style vocabulary; per-timestep reconstruction
  error for the autoencoder, its native and correct vocabulary.
- *Evasion robustness*: a black-box random-direction search for the
  smallest feature-space perturbation that flips a correctly-detected
  attack sample's prediction. This is explicitly scoped as feature-space
  robustness only, not a claim about what perturbation an attacker could
  physically achieve on the channel, since grounding the perturbation in
  the physical simulator was outside this dissertation's scope; Chapter 6
  returns to this limitation directly.

**Physical-layer depth (`experiments/phase3_physical_layer/`).** Three
further analyses using the unmodified MATLAB generator, with explicit
`rng()` seeding added on top of it (the original scripts have none):

- *SNR/difficulty sweep*: `rho_E` is swept across
  `{0.1, 0.5, 1, 2, 5, 10, 20}` with every other parameter, including
  model architecture and hyperparameters, held fixed at the Phase 1 tuned
  values. This isolates physical channel difficulty as the only variable
  under test, at the cost of also showing what happens when a fixed
  architecture is evaluated well outside the SNR regime it was tuned for
  (§ Chapter 6 discusses the MLP's low-SNR failure found this way).
- *External validation replicates*: the single-replicate external
  validation of § 3.2 is extended to five independent MATLAB runs, so
  that generalisation to unseen data is reported as a distribution rather
  than a single point estimate.
- *Regularisation ablation*: a direct follow-up on the MLP's low-SNR
  failure above, testing six configurations (baseline, two dropout
  rates, weight decay, a smaller architecture, and a smaller architecture
  combined with dropout), each retrained across five seeds at the two
  lowest-SNR sweep points, to test whether regularisation is a sufficient
  fix.

## 3.5 Reproducibility

Every script in this dissertation fixes `random_state=42` (Python) or an
explicit seed passed to MATLAB's `rng()` for its data splits and model
initialisation. This required deliberate action: the original MATLAB
generators set no seed at all, so any dataset produced without an
explicit `rng()` call is a fresh, unrepeatable Monte Carlo draw; the
`experiments/phase3_physical_layer/` scripts add explicit seeding on top
of the otherwise-unmodified original generators specifically so that the
sweep and replicate results in Chapter 5 can be regenerated. All
dependency versions are pinned in `requirements.txt`.
