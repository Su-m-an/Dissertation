# Literature Comparison

Working notes for Chapter 2 (Literature Review) and Chapter 5/6 (Results comparison
against published baselines), built from five papers supplied directly. One of them,
Hoang et al. (2021), is not just related work: it's the source this entire pipeline
was built from. The other four are physical-layer *authentication* papers, a related
but distinct problem from the *detection* task this dissertation addresses; they belong
in the literature review as context, not as a numeric comparison table.

## 1. Hoang et al. (2021): the direct source paper

**T. M. Hoang, T. Q. Duong, H. D. Tuan, S. Lambotharan, and L. Hanzo, "Physical Layer
Security: Detection of Active Eavesdropping Attacks by Support Vector Machines,"
IEEE Access, vol. 9, 2021.**

This is where `K`, `L`, `ρu`, `ρE`, the MEAN/RATIO features, the "ATD" (artificial
training data) terminology, and the TC-SVM setup all come from. The system model in
`docs/methodology.md` §1 is this paper's model, recovered independently from the
MATLAB source before this comparison was possible. Worth stating that explicitly in
the methodology chapter: the recovered model was verified against the original paper
after the fact, not copied from it.

### Direct numeric comparison

**SNR sensitivity (their Fig. 10 vs. this work's Phase 3 sweep).** Hoang et al. sweep
`ρE/ρu` from 0.1 to 1.0 at fixed `(T̂, T, T0) = (2000, 50, 15)`, `γ = 0.001`, reporting
TC-SVM accuracy rising from "about 58% at `ρE = 0.1ρu`" to "about 92% at `ρE = ρu`."
This work's Phase 3 sweep (`experiments/phase3_physical_layer/`) covers the same ratio
range at `(T̂, T, T0) = (2000, 50, 5)` with hyperparameters tuned via Phase 1's grid
search rather than fixed at `γ = 0.001`.

| `ρE/ρu` | Hoang et al., TC-SVM | This work, TC-SVM |
|---|---|---|
| 0.1 | ~58% | 59.2% |
| 1.0 | ~92% | 94.0% |

Two independent implementations, run four years apart, with different SVM
hyperparameter selection and a different `T0`, land within 1 to 2 percentage points
at both ends of the curve. This is a genuine replication, worth stating as such in
the results chapter: it's evidence the recovered physical model and the MEAN/RATIO
feature pipeline are faithful to the original, not just structurally similar.

**Baseline accuracy at the "everyday" operating point.** Hoang et al.'s Table 6, at
`T̂ = 2000, T = 50` (their closest configuration to this work's default), reports
TC-SVM accuracy of 91.28%. The corrected baseline in this repo
(`results/tc_svm_metrics.csv`) reports 94.21% (single split) and Phase 1's 5-fold CV
reports 94.25% ± 0.14%. Caveat worth stating plainly: this isn't a perfectly matched
comparison. Hoang et al.'s `T` is a single fixed window length; `ATD.csv` in this
repo aggregates every window length from `T0=5` through `T=50` into one training set
(46 window lengths × 2,000 trials × 2 classes = 184,000 rows), which is a richer
training distribution than a single-`T` set. The roughly 3-point gap is plausibly
explained by that difference plus this work's documented hyperparameter search
(Phase 1) versus Hoang et al.'s fixed `γ = 0.001`, both defensible, but not the same
experiment, and the write-up should say so rather than claim a clean improvement.

**SC-SVM.** Hoang et al. also report a single-class SVM variant (trained on normal
traffic only, no eavesdropper CSI needed) reaching up to 99% accuracy in their Table 6,
notably *higher* than their own TC-SVM. This work's closest analogue is the
autoencoder (also trained on normal sequences only, §12_autoencoder.py), which reaches
97% accuracy on the corrected baseline. Neither SC-SVM nor the autoencoder needs
eavesdropper CSI to train, which is the more realistic deployment scenario Hoang et al.
argue for in their introduction, worth a sentence in the discussion chapter tying
these two single-class approaches together as answering the same practical constraint.

### What this paper does that isn't replicated here (possible future work)

- Compares four SVM kernels (linear, RBF, polynomial, sigmoid); this work only used
  RBF, per the corrected baseline's original design choice.
- Explicitly studies the *overfitting* effect of the RBF kernel's `γ` parameter
  (their §V.E, Fig. 11), conceptually the same phenomenon this work found for the
  MLP at low SNR (`experiments/phase3_physical_layer/mlp_regularization_ablation.py`),
  but never checked for the classical models here.
- SC-SVM as a distinct trained model, rather than this work's autoencoder analogue.

## 2. The four PLA (authentication) papers: literature review context, not a comparison table

These solve a different problem: verifying that a signal really came from a claimed
legitimate transmitter (authentication), typically via an embedded tag/watermark or a
challenge-response protocol, rather than detecting whether an eavesdropper is present
at all (detection, this dissertation's task). Their metrics (probability of demodulation
error, key equivocation, false-alarm/misdetection rate for a *tag* test statistic) aren't
directly comparable to this work's accuracy/F1/AUC on eavesdropper presence. They're
genuinely useful for positioning this work within the broader physical-layer security
landscape in Chapter 2, though.

**Xie et al. (2022), "Physical Layer Authentication With High Compatibility Using an
Encoding Approach," IEEE Trans. Commun.** Tag-based authentication: a legitimate tag is
superimposed on the message, and the paper's contribution is an encoding function that
reduces how much of the source message gets modified (improving "compatibility" for
receivers unaware of the authentication scheme) without weakening robustness against
impersonation. Relevant as the tag-based branch of PLA, in contrast to this work's
energy/ML-detection branch.

**Ma et al. (2024), "Physical-Layer Authentication Enhancement via Random Watermark
Hopping," IEEE IoT Journal.** Also tag-based; alternates embedding the tag on pilot vs.
message signals via a pseudo-random sequence, trading off the security of message-tag
schemes against the low latency of pilot-tag schemes. Same tag-based family as Xie et al.

**Lu et al. (2025), "Robust Channel-Phase-Based Physical-Layer Authentication for
Multicarriers Transmission," IEEE IoT Journal.** Different mechanism again: a
challenge-response protocol using channel *phase* as the shared secret-derived test
statistic, with a specific threat model (an attacker deliberately using high transmit
power to distort the test statistic) that has no direct analogue in this work's
threat model (feature-space evasion, `experiments/phase2_depth_robustness/evasion_robustness.py`).
Worth a sentence in the evasion-robustness discussion contrasting the two threat models:
Lu et al.'s attacker manipulates transmit power to distort a *known* test statistic,
while this work's evasion experiment manipulates the *feature values themselves* to
cross a learned decision boundary. Different attack surfaces, both legitimate.

**Pan et al. (2019), "Threshold-Free Physical Layer Authentication Based on Machine
Learning for Industrial Wireless CPS," IEEE Trans. Ind. Informatics.** The closest of
the four to this work's methodology, since it's explicitly ML-based (Decision Tree, SVM,
KNN, Bagged Trees) replacing a fixed-threshold decision, evaluated on real industrial
CSI data plus field validation on USRP hardware. Directly citable in Chapter 2 as
precedent for "ML classifiers beat threshold tests," which this work's TC-SVM/RF/XGBoost
also demonstrate against the naive alternative.

One finding worth pulling into the discussion chapter: Pan et al. found that feeding
the *full-dimensional* channel matrix (8,188 features) into their classifiers performed
*worse* than a reduced 128-feature version, because "the training samples will become
sparse, which will degrade the generalization ability of the trained model." That is
essentially the same phenomenon as this work's Phase 3 finding that the MLP overfits
at low SNR when its capacity outstrips the available signal
(`experiments/phase3_physical_layer/mlp_low_snr_seed_check.py`,
`mlp_regularization_ablation.py`), two independent papers, on different tasks, hitting
the same capacity/generalization tradeoff. Worth citing together as convergent evidence,
not just a coincidence.
