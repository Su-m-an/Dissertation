# 5. Presentation of Results

*Status: full draft. Every number below is read directly from the*
*repository's result files (paths noted per table); this chapter*
*presents them, interpretation and critique is deliberately deferred to*
*Chapter 6, following the split both sample dissertations use between a*
*"presentation" and a "critical review" chapter.*

## 5.1 Baseline Model Comparison

Single stratified 80/20 train/test split, all six models, from
`results/final_model_comparison.csv`.

**Table 5.1: Baseline test-set performance.**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LSTM | 0.9925 | 0.9950 | 0.9900 | 0.9925 | 0.9996 |
| MLP | 0.9875 | 0.9899 | 0.9850 | 0.9875 | 0.9992 |
| Autoencoder | 0.9700 | 0.9498 | 0.9925 | 0.9707 | 0.9981 |
| XGBoost | 0.9423 | 0.9499 | 0.9338 | 0.9418 | 0.9799 |
| TC-SVM | 0.9421 | 0.9528 | 0.9303 | 0.9414 | 0.9789 |
| Random Forest | 0.9418 | 0.9503 | 0.9323 | 0.9412 | 0.9804 |

![Figure 5.1: Baseline accuracy, all six models.](images/fig5_1_model_accuracy_comparison.png)

**Figure 5.1.** Baseline accuracy by model, corresponding to the
Accuracy column of Table 5.1. The three sequence-based supervised models
(LSTM, MLP) and the autoencoder visibly separate from the three
classical models as a cluster.

![Figure 5.2: ROC curve, TC-SVM.](images/fig5_2_roc_curve_tc_svm.png)

**Figure 5.2.** ROC curve for TC-SVM, the classical baseline. AUC (0.979,
Table 5.1) corresponds to the area under this curve.

![Figure 5.3: Confusion matrix, TC-SVM, baseline 80/20 split.](images/fig5_3_confusion_matrix_tc_svm.png)

**Figure 5.3.** TC-SVM confusion matrix on the baseline test split. The
off-diagonal cells are the false positive / false negative counts that
Table 5.1's precision and recall are computed from.

![Figure 5.4: Precision-recall curve, TC-SVM.](images/fig5_4_precision_recall_tc_svm.png)

**Figure 5.4.** Precision-recall curve for TC-SVM at baseline class
balance (50:50). Compare against § 5.5, where the same curve's shape
degrades substantially once the class balance shifts.

### 5.1.1 Tree-Model Feature Importance

Built-in feature importance (mean decrease in impurity) from the trained
Random Forest and XGBoost models, `results/random_forest_feature_importance.csv`
and `results/xgboost_feature_importance.csv`.

![Figure 5.5: Random Forest feature importance.](images/fig5_5_rf_feature_importance.png)

![Figure 5.6: XGBoost feature importance.](images/fig5_6_xgb_feature_importance.png)

**Figures 5.5-5.6.** Both models weight `RATIO` and `MEAN` differently;
Chapter 6's SHAP analysis (§ 6.6, Figures 6.1-6.2) revisits this with a
method that also captures the *direction* of each feature's effect, not
only its relative weight.

### 5.1.2 Autoencoder Reconstruction Error Distribution

![Figure 5.7: Autoencoder reconstruction error distribution, non-attack vs. attack sequences, baseline test split.](images/fig5_7_autoencoder_reconstruction_error.png)

**Figure 5.7.** Reconstruction error distribution by true class, on the
baseline test split, after the leakage fix described in § 3.3 and
implemented in § 4.5. The calibrated threshold (95th percentile of
reconstruction error on held-out *normal* data, § 3.3.6) sits near the
upper edge of the non-attack distribution shown here; the overlap
between the two distributions is what produces the autoencoder's
false positives and false negatives reported in Table 5.1 and § 5.9.

## 5.2 Cross-Validated Performance

Stratified 5-fold CV, reported as mean ± SD, from
`experiments/phase1_statistical_rigor/results/statistical_tests_summary.json`.

**Table 5.2: 5-fold CV accuracy.**

| Model | CV Accuracy (mean ± SD) |
|---|---|
| LSTM | 0.9913 ± 0.0023 |
| MLP | 0.9897 ± 0.0021 |
| Autoencoder | 0.9625 ± 0.0062 |
| XGBoost | 0.9427 ± 0.0010 |
| TC-SVM | 0.9425 ± 0.0014 |
| Random Forest | 0.9425 ± 0.0009 |

Tuned hyperparameters (search method described in Chapter 3 § 3.4):
TC-SVM `{C: 10, gamma: 0.01}`; Random Forest
`{max_depth: 10, n_estimators: 300}`; XGBoost
`{learning_rate: 0.1, max_depth: 4, n_estimators: 100}`; MLP
`{hidden: [128, 64, 32], lr: 0.002}`; LSTM
`{hidden_size: 32, lr: 0.0005}`; Autoencoder `{latent: 4, lr: 0.001}`.

## 5.3 Statistical Significance

From the same summary file. Within-family tests use paired CV folds
(Wilcoxon) and paired test predictions (McNemar); cross-family tests use
the unpaired Mann-Whitney U (§ 3.4 explains why).

**Table 5.3: Within-family pairwise tests.**

| Group | Comparison | Wilcoxon p | McNemar p |
|---|---|---|---|
| Classical | TC-SVM vs. Random Forest | 0.813 | 0.219 |
| Classical | TC-SVM vs. XGBoost | 0.313 | 0.507 |
| Classical | Random Forest vs. XGBoost | 0.438 | 0.477 |
| Neural | MLP vs. LSTM | 0.438 | 0.219 |
| Neural | MLP vs. Autoencoder | 0.063 | 0.008 |
| Neural | LSTM vs. Autoencoder | 0.063 | 0.0001 |

**Table 5.4: Cross-family tests (Mann-Whitney U, unpaired).**

| Comparison | p-value |
|---|---|
| TC-SVM vs. MLP | 0.0109 |
| TC-SVM vs. LSTM | 0.0114 |
| TC-SVM vs. Autoencoder | 0.0117 |
| Random Forest vs. MLP | 0.0109 |
| Random Forest vs. LSTM | 0.0114 |
| Random Forest vs. Autoencoder | 0.0117 |
| XGBoost vs. MLP | 0.0112 |
| XGBoost vs. LSTM | 0.0117 |
| XGBoost vs. Autoencoder | 0.0119 |

## 5.4 Calibration, Latency, and Model Size

From `experiments/phase1_statistical_rigor/results/calibration_latency_summary.csv`.

**Table 5.5: PR-AUC, inference cost, and size.**

| Model | PR-AUC | Latency (ms/sample) | Model size (bytes) | Params |
|---|---|---|---|---|
| LSTM | 0.9998 | 0.0105 | 55,989 | 12,961 |
| Autoencoder | 0.9983 | 0.00012 | 23,027 | 4,502 |
| MLP | 0.9992 | 0.00014 | 71,209 | 16,897 |
| XGBoost | 0.9836 | 0.00015 | 176,301 | n/a (100 trees) |
| Random Forest | 0.9832 | 0.00206 | 24,941,945 | n/a (300 trees) |
| TC-SVM | 0.9817 | 0.377 | 839,579 | n/a (23,276 support vectors) |

![Figure 5.8: Reliability diagrams, all six models.](images/fig5_8_reliability_diagrams.png)

**Figure 5.8.** Reliability (calibration) diagrams: predicted probability
on the x-axis, observed frequency of the positive class on the y-axis,
per model. A perfectly calibrated model traces the diagonal; deviations
above or below indicate under- or over-confidence at that probability
range. This is the figure the PR-AUC column of Table 5.5 is a scalar
summary of.

## 5.5 Class Imbalance Stress Test

From `experiments/phase2_depth_robustness/results/class_imbalance_stress_test.csv`.
Precision/recall/F1/PR-AUC, not accuracy, are the relevant columns here
(§ 3.4 explains why); full table has all six models at 90:10 and 95:5,
classical models additionally at 99:1 and 99.9:0.1 (sequence models were
not evaluated at these last two ratios: too few positive test samples
remained at the sequence dataset's smaller size to report a stable
metric, and are correctly marked as skipped in the results file rather
than reported on an unreliable sample).

**Table 5.6: Selected class-imbalance results.**

| Model | Ratio | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| TC-SVM | 90:10 | 0.904 | 0.803 | 0.851 | 0.915 |
| TC-SVM | 99:1 | 0.713 | 0.306 | 0.429 | 0.553 |
| TC-SVM | 99.9:0.1 | 0.000 | 0.000 | 0.000 | 0.005 |
| Random Forest | 90:10 | 0.890 | 0.825 | 0.857 | 0.917 |
| Random Forest | 99.9:0.1 | 0.000 | 0.000 | 0.000 | 0.117 |
| XGBoost | 90:10 | 0.891 | 0.827 | 0.858 | 0.919 |
| MLP | 90:10 | 0.933 | 0.955 | 0.944 | 0.997 |
| LSTM | 90:10 | 0.956 | 0.977 | 0.966 | 0.999 |
| Autoencoder | 90:10 | 0.657 | 1.000 | 0.793 | 0.997 |
| Autoencoder | 95:5 | 0.500 | 1.000 | 0.667 | 0.998 |

## 5.6 Representation and Sequence-Length Ablation

From `experiments/phase2_depth_robustness/results/representation_sequence_ablation.csv`.
Model capacity held fixed; only the input representation varies.

**Table 5.7: Accuracy vs. representation.**

| Representation | n features | MLP accuracy | MLP AUC | LSTM accuracy | LSTM AUC |
|---|---|---|---|---|---|
| MEAN only | 1 | 0.990 | 0.9998 | n/a | n/a |
| MEAN+RATIO | 2 | 0.991 | 0.9998 | n/a | n/a |
| E1-E5 | 5 | 0.770 | 0.852 | 0.776 | 0.853 |
| E1-E10 | 10 | 0.859 | 0.934 | 0.869 | 0.937 |
| E1-E20 | 20 | 0.935 | 0.983 | 0.944 | 0.986 |
| E1-E30 | 30 | 0.958 | 0.990 | 0.964 | 0.994 |
| E1-E40 | 40 | 0.979 | 0.999 | 0.983 | 0.997 |
| E1-E50 (full) | 50 | 0.989 | 0.999 | 0.993 | 0.9998 |

![Figure 5.9: MLP accuracy vs. representation, from a single MEAN statistic through the full 50-step sequence.](images/fig5_9_representation_ladder.png)

**Figure 5.9.** The "representation richness ladder": a fixed, compact
MLP's accuracy as the input representation grows from `MEAN` alone
through `E1-E50`. The sharp drop from `MEAN`/`MEAN+RATIO` to `E1-E5`,
followed by a steady climb back up as more of the sequence is added, is
the shape Table 5.7's numbers describe; visualised, it reads as a
distinct V-shape rather than a monotonic trend.

![Figure 5.10: MLP vs. LSTM accuracy at each fixed sequence length.](images/fig5_10_sequence_length_comparison.png)

**Figure 5.10.** The same representation ladder, with MLP and LSTM
plotted together at each truncated sequence length. The LSTM's curve
sits consistently above the MLP's at every point, the small,
consistent architectural gap referenced in § 6.2.

## 5.7 Physical-Layer SNR Sweep

`rho_E` swept from 0.1 to 20 (`rho_u=5` fixed), all six models at Phase 1
tuned hyperparameters, from
`experiments/phase3_physical_layer/results/snr_sweep_results.csv`.

**Table 5.8: Accuracy vs. rho_E.**

| rho_E | TC-SVM | Random Forest | XGBoost | MLP | LSTM | Autoencoder |
|---|---|---|---|---|---|---|
| 0.1 | 0.522 | 0.519 | 0.516 | 0.194 | 0.515 | 0.493 |
| 0.5 | 0.592 | 0.594 | 0.594 | 0.435 | 0.608 | 0.519 |
| 1.0 | 0.671 | 0.673 | 0.674 | 0.651 | 0.716 | 0.563 |
| 2.0 | 0.793 | 0.794 | 0.795 | 0.831 | 0.868 | 0.698 |
| 5.0 (baseline) | 0.940 | 0.942 | 0.941 | 0.993 | 0.998 | 0.975 |
| 10.0 | 0.984 | 0.984 | 0.985 | 1.000 | 1.000 | 0.981 |
| 20.0 | 0.996 | 0.996 | 0.997 | 1.000 | 1.000 | 0.985 |

**Table 5.9: MLP low-SNR seed check** (5 seeds each,
`experiments/phase3_physical_layer/results/mlp_low_snr_seed_check.csv`),
run to check whether the MLP's accuracy at `rho_E=0.1` (below TC-SVM's,
despite outperforming it everywhere else in Table 5.8) is a single-seed
artefact.

| rho_E | Accuracy (mean ± SD, 5 seeds) | AUC (mean ± SD, 5 seeds) |
|---|---|---|
| 0.1 | 0.191 ± 0.008 | 0.1058 ± 0.0058 |
| 0.5 | 0.426 ± 0.011 | 0.4136 ± 0.0058 |

![Figure 5.11: Accuracy vs. rho_E, all six models.](images/fig5_11_snr_sweep_accuracy.png)

**Figure 5.11.** Accuracy across the SNR sweep, Table 5.8 plotted. The
MLP's curve is visibly the outlier at the lowest two `rho_E` points,
falling below every classical model, before crossing back above all of
them from `rho_E=2` onward.

![Figure 5.12: AUC vs. rho_E, all six models.](images/fig5_12_snr_sweep_auc.png)

**Figure 5.12.** The same sweep measured by AUC rather than accuracy.
The MLP's AUC at `rho_E=0.1` (0.106) sits *below* 0.5, worse than a
random classifier, a stronger and more precise statement than the
accuracy figure alone, since AUC is threshold-independent, this isn't an
artefact of a badly-placed decision threshold.

## 5.8 Regularisation Ablation (Follow-up on MLP Low-SNR Result)

Six configurations, 5 seeds each, at the two lowest `rho_E` sweep points,
from
`experiments/phase3_physical_layer/results/mlp_regularization_ablation_summary.csv`.

**Table 5.10: Train/test AUC gap by configuration, rho_E=0.1.**

| Config | Train AUC | Test AUC | Gap |
|---|---|---|---|
| baseline | 0.748 | 0.104 | 0.644 |
| dropout_0.3 | 0.657 | 0.180 | 0.477 |
| dropout_0.5 | 0.594 | 0.310 | 0.283 |
| smaller | 0.734 | 0.141 | 0.593 |
| smaller_dropout | 0.625 | 0.252 | 0.372 |
| weight_decay | 0.583 | 0.349 | 0.235 |

**Table 5.11: Train/test AUC gap by configuration, rho_E=0.5.**

| Config | Train AUC | Test AUC | Gap |
|---|---|---|---|
| baseline | 0.998 | 0.414 | 0.584 |
| dropout_0.3 | 0.887 | 0.477 | 0.410 |
| dropout_0.5 | 0.796 | 0.535 | 0.261 |
| smaller | 0.909 | 0.433 | 0.476 |
| smaller_dropout | 0.789 | 0.553 | 0.236 |
| weight_decay | 0.954 | 0.452 | 0.502 |

![Figure 5.13: Test AUC by regularisation configuration, both SNR points, with 5-seed error bars.](images/fig5_14_mlp_regularization_ablation.png)

**Figure 5.13.** Test AUC (not the train/test gap of Tables 5.10/5.11,
a complementary view) by configuration. The dashed line marks chance-level
AUC (0.5); at `rho_E=0.1` no configuration reaches it, at `rho_E=0.5`
`smaller_dropout` and `dropout_0.5` do. Error bars are one standard
deviation across the 5 seeds, small relative to the between-configuration
differences, supporting that the ranking between configurations is not
seed noise.

## 5.9 Error Analysis

From `experiments/phase2_depth_robustness/results/error_analysis_summary.csv`,
using each model's Phase 1 saved test-set predictions.

**Table 5.12: Error counts.**

| Model | Errors | False positives | False negatives |
|---|---|---|---|
| TC-SVM | 2,100 | 786 | 1,314 |
| Random Forest | 2,079 | 842 | 1,237 |
| XGBoost | 2,089 | 867 | 1,222 |
| MLP | 10 | 4 | 6 |
| LSTM | 6 | 3 | 3 |
| Autoencoder | 22 | 18 | 4 |

For the autoencoder, mean reconstruction error is 1.41 on true positives
(correctly flagged attacks) versus 4.52 on false negatives (missed
attacks): missed attacks reconstruct comparably well to normal traffic,
consistent with them being borderline cases rather than model errors on
clearly-anomalous input.

![Figure 5.14: TC-SVM/Random Forest/XGBoost errors plotted in MEAN-RATIO feature space.](images/fig5_15_error_analysis_classical.png)

**Figure 5.14.** Classical-model errors (Table 5.12) plotted in
`[MEAN, RATIO]` space. Errors visibly cluster near the class boundary
region visible in Figure 4.4, consistent with these being borderline
cases rather than errors scattered arbitrarily across feature space.

![Figure 5.15: MLP/LSTM predicted probability distribution across the full baseline test set, correct vs. misclassified.](images/fig5_16_error_analysis_neural.png)

**Figure 5.15.** Predicted probability of "attack" across the full
baseline test set for the MLP and LSTM, correct predictions in blue,
the handful of misclassifications (Table 5.12: 10 and 6 respectively) in
red. Both models are confidently bimodal, concentrated near 0 or 1 with
very little mass near the 0.5 decision boundary, and the (rare)
misclassifications are correspondingly hard to see against that mass at
this scale.

![Figure 5.16: Autoencoder reconstruction error, true positives vs. false negatives.](images/fig5_17_error_analysis_autoencoder.png)

**Figure 5.16.** The 1.41 vs. 4.52 mean reconstruction error gap quoted
above, shown as a distribution rather than two summary numbers.

## 5.10 Interpretability

- **SHAP** (TreeExplainer, Random Forest and XGBoost): full plots in
  `experiments/phase2_depth_robustness/`, summarised in Chapter 6.
- **LSTM** (`experiments/phase2_depth_robustness/results/lstm_timestep_importance.csv`):
  timestep occlusion, permutation importance, and Integrated Gradients
  each independently rank timesteps 36-46 as most important to the
  LSTM's decision.
- **Autoencoder**
  (`experiments/phase2_depth_robustness/results/autoencoder_timestep_reconstruction_error.csv`):
  the single worst-reconstructed timestep under an attack sequence is
  timestep 46, within the same window the LSTM independently identifies
  as most important.

## 5.11 Evasion Robustness

Black-box, feature-space random-direction search for the smallest
perturbation (in standard deviations of the relevant feature) that flips
a correctly-detected attack sample's prediction, from
`experiments/phase2_depth_robustness/results/evasion_robustness_feature_space.csv`.

**Table 5.13: Feature-space evasion distance.**

| Model | Median min. distance (SD) | % evaded within 2 SD | % never evaded within 5 SD |
|---|---|---|---|
| TC-SVM | 1.00 | 92.0% | 0.0% |
| Random Forest | 1.00 | 96.0% | 0.0% |
| XGBoost | 1.00 | 91.3% | 0.0% |
| Autoencoder | 1.75 | 2.0% | 97.3% |
| LSTM | 3.50 | 4.7% | 75.3% |
| MLP | 4.13 | 2.7% | 81.3% |

## 5.12 External Validation Replicates

Classical models (Phase 1 saved weights, no retraining) scored on five
independent MATLAB simulation runs, from
`experiments/phase3_physical_layer/results/external_validation_replicates_summary.csv`.

**Table 5.14: Accuracy across 5 independent replicates.**

| Model | Accuracy (mean ± SD) | AUC (mean ± SD) |
|---|---|---|
| XGBoost | 0.9384 ± 0.0025 | 0.9796 ± 0.0009 |
| Random Forest | 0.9381 ± 0.0023 | 0.9795 ± 0.0009 |
| TC-SVM | 0.9380 ± 0.0025 | 0.9778 ± 0.0011 |

![Figure 5.17: Internal test-split accuracy vs. external validation accuracy, all three classical models, single-replicate comparison.](images/fig5_13_external_validation_comparison.png)

**Figure 5.17.** The original single-replicate external validation
result (`src/14_external_validation.py`, § 3.2/4.4: models trained on
`ATD.csv`, evaluated with no retraining on the independently-generated
`ATD_features.csv`), internal test-split accuracy against external
accuracy for all three classical models. The bars are visually almost
identical per model, the gap is under half a percentage point throughout,
which is the single-replicate version of the five-replicate result
summarised numerically in Table 5.14 above; a five-bar-group version of
this figure would be difficult to read at this scale, which is why Table
5.14 reports that extension as summary statistics rather than a second
figure.
