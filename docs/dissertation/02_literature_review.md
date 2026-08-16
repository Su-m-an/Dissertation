# 2. Literature Review

*Status: scaffold. Five papers are available in full detail (see*
*`docs/literature_comparison.md` for the working notes this is drawn*
*from); you may want more before this chapter is submission-ready,*
*both samples cite considerably more than five sources. Structure below*
*follows the pattern in both samples: an overview of the problem space,*
*then organised by technique/approach family, then a section that*
*explicitly identifies the gap this dissertation fills.*

## 2.1 Physical-Layer Security: Overview

- Position physical-layer security (PLS) as a field: securing wireless
  communication using properties of the channel itself, rather than
  (or alongside) cryptographic protocols at higher layers.
- Two branches worth distinguishing explicitly, since the four
  non-source papers below split across them:
  - **Authentication**: verifying a received signal really came from
    the claimed legitimate transmitter (tag-embedding, challenge-response).
  - **Detection**: deciding whether an attacker is present at all,
    this dissertation's task, and Hoang et al.'s.
- One or two sentences on why detection and authentication are related
  but distinct problems with different evaluation metrics (detection:
  accuracy/F1/AUC on attacker presence; authentication: demodulation
  error rate, key equivocation, tag false-alarm/misdetection rate).

## 2.2 Pilot Contamination as an Attack Vector

- Hoang et al. (2021) [1], the direct source of this work's system
  model, features, and TC-SVM baseline. Cover in detail:
  - Their system model (`K` users, `L` antennas, MMSE channel
    estimation, pilot contamination mechanism).
  - Their MEAN/RATIO feature construction and why it's a natural
    reduction of the received-power sequence.
  - Their reported results (TC-SVM ~91-92% accuracy at their default
    operating point, SC-SVM up to ~99% using only legitimate-traffic
    training data, SNR-sensitivity curve from ~58% at `ρE=0.1ρu` to
    ~92% at `ρE=ρu`).
  - What they don't do that this work extends: only RBF kernel is
    used here versus their four-kernel comparison (worth noting as a
    scope limitation); they don't try sequence models.
  - Full detail and the direct numeric comparison against this work's
    results is in `docs/literature_comparison.md` §1, useful sourcs to pull
    the comparison table into Chapter 5/6 from.
  - Frame their detection task formally here, worth a sentence or two
    before Chapter 3 develops it in full: pilot contamination detection
    reduces to a binary hypothesis test (attacker present vs. absent)
    where the optimal, model-based decision rule (a likelihood-ratio
    test) is unavailable in practice because it requires knowing the
    eavesdropper's own transmit power and channel statistics, information
    the legitimate receiver by definition does not have. This is *why*
    Hoang et al., and this dissertation, turn to a data-driven classifier
    rather than deriving a closed-form detector; worth stating explicitly
    here since it's the conceptual justification for the entire
    machine-learning approach, not just an implementation detail.

## 2.3 Physical-Layer Authentication: Related but Distinct Work

Four papers, useful for positioning this work within the broader PLS
landscape even though their metrics aren't directly comparable. Full
notes in `docs/literature_comparison.md` §2.

- **Xie et al. (2022)** [2] and **Ma et al. (2024)** [3]: tag-based
  authentication (embedding a watermark on the message or pilot signal).
  Group these together as the "tag-based" family.
- **Lu et al. (2025)** [4]: challenge-response authentication using
  channel phase as the shared secret-derived statistic. Worth a
  forward-reference here to the evasion-robustness discussion in
  Chapter 6: their threat model (attacker manipulates transmit power to
  distort a *known* test statistic) contrasts usefully with this work's
  evasion experiment (attacker manipulates *feature values* directly to
  cross a *learned* decision boundary).
- **Pan et al. (2019)** [5]: the closest of the four methodologically,
  since it's explicitly ML-based (Decision Tree, SVM, KNN, Bagged
  Trees replacing a fixed threshold), validated on real industrial CSI
  data and USRP hardware. Cite as precedent for "learned classifiers
  beat fixed-threshold tests," which this work's classical baseline
  also demonstrates. Their finding that full-dimensional features
  (8,188-dim) generalise *worse* than a reduced 128-dim version is
  worth flagging here as a forward-reference to Chapter 6, where this
  work finds an analogous capacity/generalisation failure in the MLP at
  low SNR.

## 2.4 Machine Learning for Wireless Anomaly/Intrusion Detection

*Scope note: this section isn't populated from any paper supplied so*
*far, it's a gap. A literature review this size (both samples run*
*10-14 pages) will likely need it: broader ML-for-wireless-security*
*work (not physical-layer specific) to justify the six-model comparison*
*(SVM/RF/XGBoost/MLP/Autoencoder/LSTM) as a reasonable design choice, and*
*prior use of autoencoders for anomaly detection outside this specific*
*application, to justify why the autoencoder's single-class formulation*
*was included at all. Worth a literature search pass before writing this.*

## 2.5 Summary and Positioning of This Work

- Explicitly state what's missing from the reviewed literature that this
  dissertation addresses: no reviewed detection paper compares a
  hand-engineered-feature baseline against sequence-consuming deep
  learning models on the same task and data; Hoang et al. establish the
  baseline this work extends but stop at feature engineering plus SVM
  variants.
- Set up the transition into Chapter 3: "This dissertation adopts Hoang
  et al.'s system model and classical baseline as a validated starting
  point, and extends it with..." (methodology chapter follows).

## References for this chapter

Uses citations `[1]`-`[5]`, see `09_references.md`. Add more as the
literature search in §2.4 turns up further sources, renumber
consistently across the whole document if so.
