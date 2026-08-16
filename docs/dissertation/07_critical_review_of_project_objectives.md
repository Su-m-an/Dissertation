# 7. Critical Review of the Project Objectives

*Status: scaffold, and more so than any other chapter, this one should be*
*written in your own first-person voice: it's a personal assessment of*
*your own work against your own stated goals, not a technical summary.*
*What's below for each objective is the factual basis (what was actually*
*done, traceable to Chapters 4-6), not a drafted verdict; write the*
*"met / partially met / not met" judgement and the reflection around it*
*yourself. Both samples this format is drawn from are candid here,*
*Sample 8 plainly states an evaluation method was never completed rather*
*than implying otherwise, that kind of honesty reads as strength, not*
*weakness, in this chapter specifically.*

Recall the six objectives from § 1.3.

## 7.1 Objective 1: Reproduce and Validate the Classical Baseline

What was done: TC-SVM, Random Forest, and XGBoost implemented on the
`MEAN`/`RATIO` features from Hoang et al.'s framework (§ 3.1-3.3); the
SNR-sensitivity sweep independently reproduces their Fig. 10 within 1-2
points at both ends (§ 6.8); baseline accuracy is within ~3 points of
their reported figure, with a stated, plausible (not proven) explanation
for the gap.

What wasn't done: only the RBF kernel was used; Hoang et al. compare four
kernels (linear, RBF, polynomial, sigmoid), and their own finding about
`gamma`-driven overfitting (§ 2.2) was never checked against this work's
classical models specifically, only observed analogously in the MLP.

*Your assessment: met / partially met / not met, and why.*

## 7.2 Objective 2: Sequence-Consuming Deep Learning vs. Classical Baseline

What was done: MLP, LSTM, and Autoencoder implemented on the raw
50-step sequence and compared against the classical baseline on
accuracy, F1, AUC, with cross-validation and significance testing.

What's complicated, not simply achieved: § 6.1 is direct about this, the
significance evidence for "sequence models beat classical models" rests
on the weakest test used in this dissertation (unpaired Mann-Whitney,
different underlying datasets), and § 6.2's ablation shows the advantage
isn't really about "raw sequence vs. hand-engineered features" so much as
"the full sequence eventually recovers what a well-chosen 2-feature
summary already captured." Worth deciding, in your own words, whether
that nuance means this objective was fully met, or only partially,
depending on how strictly "compare their detection performance" is read.

*Your assessment.*

## 7.3 Objective 3: Statistically Rigorous Evaluation Protocol

What was done: stratified 5-fold CV with documented hyperparameter
search for all six models; Wilcoxon signed-rank and McNemar within model
families; Mann-Whitney across families, correctly labelled as weaker
evidence rather than presented as equivalent; calibration curves,
PR-AUC, latency, and model size for every model.

What wasn't done: a paired cross-family test (would require evaluating
classical models on the sequence dataset or vice versa, § 6.1, named as
a limitation rather than attempted).

*Your assessment, this one is closest to straightforwardly met.*

## 7.4 Objective 4: Robustness Under Realistic Deployment Conditions

What was done: class imbalance down to 99.9:0.1 (§ 5.5), a `rho_E` SNR
sweep across two orders of magnitude (§ 5.7), and black-box feature-space
evasion search (§ 5.11).

What's limited: the evasion result has two named, unresolved caveats
(feature-space only, not physically grounded; a dimensionality confound
between the 2-feature and 50-feature models, § 6.7), the class-imbalance
result correctly stops reporting at the point where too few positive
samples remain rather than extrapolating past it.

*Your assessment.*

## 7.5 Objective 5: Ablation and Interpretability

What was done: the representation/sequence-length ablation directly
answers "architecture vs. information" (§ 6.2, LSTM has a small but
consistent edge at every fixed sequence length, but most of the total
performance comes from information, not architecture); SHAP, LSTM
occlusion/permutation/Integrated Gradients, and autoencoder
reconstruction error all implemented, with a genuinely convergent finding
(timesteps 36-46, four independent methods, two unrelated models, § 6.6).

What's open: § 6.6 names a physical question (why that specific window)
that this dissertation raises but does not resolve. Worth deciding
whether "investigate... and identify which parts of the input sequence
drive their decisions" is met by identifying the window without
explaining the underlying physical cause.

*Your assessment.*

## 7.6 Objective 6: Deployment Trade-offs

What was done: latency, model size, and parameter count for all six
models (§ 5.4/6.4); the single-class (autoencoder) vs. supervised
trade-off discussed directly and tied to Hoang et al.'s SC-SVM (§ 6.3).

*Your assessment, this one is also close to straightforwardly met.*

## 7.7 Overall

*Write a short closing paragraph here: on balance, how much of the stated*
*aim (§ 1.3) was achieved? Where the honest answer is "partially," say so*
*and say why, per-objective detail above gives you the material, this*
*paragraph is where you weigh it. This is also a natural place to note*
*anything you'd have scoped differently with hindsight, that's exactly*
*the kind of reflection both samples include here and nowhere else in*
*the document.*
