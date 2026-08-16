# 6. Critical Review of the Results

*Status: scaffold. This is where your own analytical voice matters most,*
*the raw findings and their implications are laid out below, but turning*
*them into an argument, deciding what's most important, what qualifies*
*what, is the actual intellectual work of this chapter. Both samples*
*give this chapter named subsections addressing specific*
*questions/limitations rather than working through results in table*
*order, that pattern is followed below.*

## 6.1 Is the Sequence-Based Advantage Real?

The headline comparison (Table 5.1/5.2) shows the LSTM ahead of the
classical baseline by roughly 5 points of accuracy. Whether that's a
meaningful claim depends entirely on the significance tests in § 5.3, not
the point estimates in § 5.1/5.2 alone, points to make:

- Within the classical family, none of the pairwise differences are
  significant (all Wilcoxon p > 0.3), the three classical models are
  statistically indistinguishable from each other. Worth stating plainly:
  this dissertation cannot claim XGBoost beats TC-SVM, or vice versa.
- Within the neural family, MLP vs. LSTM is also not significant
  (Wilcoxon p = 0.44), only Autoencoder vs. {MLP, LSTM} reaches
  significance (McNemar p < 0.01). The LSTM's numerical edge over the MLP
  in Table 5.1 should not be oversold as a demonstrated architectural
  advantage on this evidence alone.
- The cross-family Mann-Whitney tests (all p < 0.02) are the only
  evidence that sequence models beat the classical baseline, and they're
  the *weakest* form of evidence used in this dissertation (unpaired,
  across different underlying datasets, § 3.4). This is worth stating as
  a limitation rather than glossing over: the central claim of this
  dissertation, sequence representation beats feature engineering, rests
  on the weakest statistical tool available here, not the strongest.
- What would strengthen this claim, and is worth naming explicitly as a
  gap: evaluating classical models on the sequence dataset directly (or
  vice versa) so a *paired* test across families becomes possible. This
  wasn't done and is a legitimate limitation, name it, don't hide it.

## 6.2 Where Does the Advantage Come From? (Representation Ablation)

Table 5.7 is the more informative result than the headline table, points
to draw out:

- `MEAN` alone (a *single* number) achieves 0.99 accuracy with an MLP,
  matching the full 50-step sequence and *beating* every intermediate
  truncated-sequence length (E1-E5 through E1-E40). This is a striking,
  slightly counterintuitive result worth dwelling on: more raw
  information (a longer partial sequence) is worse than the right
  two-number summary, until the sequence is long enough to reconstruct
  what the summary statistic already captured.
- This substantially qualifies § 6.1's framing. The advantage of
  "sequence-based deep learning" isn't from access to raw information
  per se, `MEAN` alone already contains most of what's needed, it's from
  the *full* sequence eventually recovering (and, per the LSTM's edge,
  perhaps modestly exceeding) what a well-chosen hand-engineered feature
  already provides. Worth a direct sentence reconciling this with the
  Introduction's framing (§ 1.2) of "does the full sequence recover
  information the summary statistic discards": the answer this ablation
  gives is "yes, but the summary statistic wasn't discarding much to
  begin with, in this system model."
- At every truncated sequence length, LSTM beats MLP by a small but
  consistent margin (e.g. 0.869 vs. 0.859 at E1-E10), suggesting
  *some* genuine architectural value from modelling order, separate from
  total information content. Consistent with, and a partial resolution
  of, the ambiguity flagged in § 6.1.

## 6.3 The Autoencoder: a Different Value Proposition, Not a Worse Model

- The autoencoder trails MLP/LSTM on accuracy (0.97 vs. 0.99) but is
  trained *without ever seeing an attack example* (§ 3.3), a materially
  different, and arguably more realistic, deployment assumption: labelled
  attack data may not be available in practice.
- This is the same trade-off Hoang et al.'s SC-SVM makes (§ 2.2,
  `docs/literature_comparison.md` § 1), worth an explicit tie-back here:
  two independent single-class approaches, on the same underlying
  problem, both trading a few points of accuracy for not needing
  adversarial training data.
- At severe class imbalance (Table 5.6), the autoencoder is the *only*
  model with recall = 1.0 at both 90:10 and 95:5, at the cost of much
  lower precision (0.50-0.66). Whether that trade-off (catch everything,
  tolerate more false alarms) is the right one depends on the deployment
  context, worth a sentence on when it would and wouldn't be.

## 6.4 Deployment Cost, Not Just Accuracy

Table 5.5, worth foregrounding rather than treating as a footnote:

- TC-SVM's inference latency (0.377 ms/sample) is roughly 30-3,000x
  slower than every other model, and its model size (839 KB, 23,276
  support vectors) is non-trivial for an edge deployment. This matters
  more than its accuracy rank might suggest for a system that has to run
  in real time at a base station.
- Random Forest's model size (25 MB, 300 trees) is by far the largest of
  the six despite middling accuracy, worth flagging as the weakest model
  on a size-adjusted basis.
- The autoencoder is both the smallest model (23 KB) and near the top of
  the PR-AUC ranking (0.998), worth stating as a genuinely strong
  practical case for it independent of the single-class argument in
  § 6.3.

## 6.5 The MLP's Low-SNR Failure, and What Regularisation Does About It

- At `rho_E=0.1` (Table 5.8), the MLP scores *below chance-adjacent*
  (0.194 accuracy, 0.106 AUC) while every classical model still manages
  ~0.52. This is not a fluke: the 5-seed check (Table 5.9) confirms it's
  systematic (0.191 ± 0.008 accuracy across seeds).
- Diagnosis: comparing train vs. test AUC directly at this point shows a
  large gap (Table 5.10, baseline: train 0.748, test 0.104), the model is
  overfitting, not simply performing poorly. Worth stating why this
  matters methodologically: this wasn't assumed from the low test score
  alone, it was verified via the train/test comparison, "which is
  overfitting, not just a hard problem" is a claim this dissertation can
  actually support with evidence.
- The regularisation ablation (Tables 5.10/5.11) shows dropout does most
  of the work (dropout_0.5 alone nearly halves the gap at both SNR
  points), weight decay helps more at `rho_E=0.1` than architecture size
  does, and a smaller architecture *alone*, without dropout, barely
  helps (0.644 to 0.593 gap, versus 0.644 to 0.283 with dropout_0.5).
  Worth a direct claim: capacity alone isn't the problem, or dropout
  wouldn't outperform simply shrinking the network.
- Regularisation *narrows* but does not *fully close* the gap at either
  point, at `rho_E=0.1` even the best configuration (weight_decay,
  gap 0.235) still leaves test AUC at 0.349, barely above chance. Report
  this as a partial, not complete, fix, both samples this format is
  drawn from are explicit about incomplete fixes rather than implying
  success beyond what the numbers support.
- Tie to Pan et al. (2019) [5] here (`docs/literature_comparison.md` § 2,
  final paragraph): their full-dimensional-feature overfitting finding on
  a completely different task is the same capacity/generalisation
  phenomenon found here, worth citing as convergent evidence across two
  independent studies, not coincidence.

## 6.6 Interpretability: Convergent Evidence for a Specific Time Window

**Tree models: SHAP.**

![Figure 6.1: SHAP summary plot, Random Forest.](images/fig6_1_shap_random_forest.png)

![Figure 6.2: SHAP summary plot, XGBoost.](images/fig6_2_shap_xgboost.png)

**Figures 6.1-6.2.** Per-sample SHAP values for `MEAN` and `RATIO`, both
models. Unlike the impurity-based importance in Figures 5.5-5.6, SHAP
additionally shows *direction*: which feature values push a prediction
toward "attack" versus "non-attack," and by how much for each individual
sample, not just an aggregate weight. Worth writing a paragraph
comparing what these two figures say against Figures 5.5/5.6, do the
two importance methods agree on which feature dominates, or not, and
what would it mean for this dissertation's claims if they disagreed?

**LSTM and autoencoder: timestep-level evidence.**

![Figure 6.3: LSTM timestep importance (occlusion, permutation, Integrated Gradients).](images/fig6_3_lstm_timestep_importance.png)

![Figure 6.4: Autoencoder per-timestep reconstruction error under attack sequences.](images/fig6_4_autoencoder_timestep_reconstruction.png)

**Figures 6.3-6.4.** Three independent LSTM interpretability methods
(occlusion, permutation, Integrated Gradients, Figure 6.3) and the
autoencoder's per-timestep reconstruction error (Figure 6.4, an entirely
different model, different mechanism) all separately point to timesteps
36-46 as most important (§ 5.10). Worth stating why this convergence is
meaningful: these are four different methods on two architecturally
unrelated models, this isn't one method's artefact, and the two figures
above are the direct visual evidence for that claim, worth describing
what specifically overlaps between them (do the peaks in Figure 6.3 line
up with the peak in Figure 6.4 within a timestep or two, or only
approximately?) rather than only citing the summary numbers in § 5.10.

- Open question worth naming, not necessarily answering: is there a
  physical reason timesteps 36-46 specifically matter, e.g. does the
  Rayleigh fading process or the accumulation of `RATIO`'s excess-power
  statistic make later timesteps disproportionately informative? This
  connects back to the `MEAN`-alone result in § 6.2, if information
  accumulates roughly uniformly over the window, why would specific late
  timesteps dominate the LSTM's attention? Worth exploring or at least
  flagging as future work if not resolved here.

## 6.7 Evasion Robustness: What This Result Does and Doesn't Show

![Figure 6.5: Distribution of minimum evasion distance (standard deviations), all six models.](images/fig6_5_evasion_robustness.png)

**Figure 6.5.** Distribution underlying Table 5.13: for each model, the
smallest feature-space perturbation (in SD) that flipped a correctly-
detected attack sample's prediction. The two-cluster pattern (classical
models bunched near 1 SD, deep learning models spread from 1.75-4+ SD) is
visible directly here, worth describing whether the two clusters overlap
at all, since a claim of "clean separation" is stronger than "mostly
separated, with some overlap," and the two limitations below determine
how much weight that separation, however clean, can actually bear.

- Table 5.13 shows classical models are evaded within a median 1
  standard deviation of feature perturbation, versus 1.75-4.1 SD for the
  deep learning models, a large apparent robustness gap.
- Two limitations that should qualify this claim heavily, both already
  flagged in the methodology (§ 3.4) and worth repeating here with
  teeth: (1) this is feature-space robustness, not a claim that an
  attacker could physically achieve this perturbation on the channel,
  translating a "distance in `MEAN`/`RATIO`-space" or "distance in
  `E1`-`E50` space" into an actual achievable `rho_E`/fading realisation
  was out of scope. (2) the classical and deep learning models are
  perturbed in different-dimensional spaces (2 features vs. 50), and
  distance in standard deviations is not obviously comparable across
  dimensionality, part of the apparent robustness gap could be a
  dimensionality artefact rather than a genuine robustness difference.
  State this as an open confound, not a resolved finding.

## 6.8 Comparison Against Hoang et al. (2021)

Full detail in `docs/literature_comparison.md` § 1; pull the direct
numeric comparison table into this section. Key points:

- The SNR-sensitivity replication (their Fig. 10 vs. this work's Table
  5.8) lands within 1-2 percentage points at both ends of the curve
  despite two independent implementations four years apart, worth
  stating as genuine validation of the recovered system model (§ 3.1),
  not just a structural resemblance.
- The baseline accuracy gap (94.25% here vs. their reported 91.28%) is
  plausibly explained by a richer training distribution (this work
  trains across all window lengths `T0`-`T` at once) and a documented
  hyperparameter search versus their fixed `gamma`, state this as a
  plausible explanation, not a demonstrated one, since the two setups
  aren't a controlled comparison.

## 6.9 Overall Limitations of This Work

Consolidate from the points above into a short, direct list, this is
where the "critical" in critical review earns its name:

- Cross-family significance rests on the weakest available test (§ 6.1).
- The representation ablation complicates rather than confirms the
  dissertation's own framing of its central contribution (§ 6.2).
- Evasion robustness has an unresolved dimensionality confound (§ 6.7).
- The MLP's low-SNR failure is only partially fixed (§ 6.5).
- Everything in this dissertation is evaluated on simulated data from
  one specific system model (`K=4, L=10`, Rayleigh fading); no claim is
  made, or should be made, about real hardware or a different system
  configuration without further validation.
