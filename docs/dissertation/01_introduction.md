# 1. Introduction

*Status: scaffold. This chapter should be your own scholarly framing, the*
*points, facts, and structure below are the raw material; the prose is*
*for you to write. The one exception is the Aims and Objectives list,*
*which is drafted in full below since Chapter 7 is assessed directly*
*against it, if you change it here, mirror the change there.*

## 1.1 Background

Points to cover, in roughly this order:

- What massive MIMO is and why pilot-based channel estimation is central
  to it (base station needs channel state information to serve `K` users
  over `L` antennas; pilots are how it gets that).
- Why pilots are a weak point: they're typically public/known sequences,
  reused across cells for scalability, which opens the door to *pilot
  contamination*.
- Distinguish the two things "pilot contamination" can mean in the
  literature: (a) unintentional contamination from frequency reuse
  between cells, a capacity/estimation-accuracy problem, versus (b)
  deliberate, adversarial contamination, an active eavesdropper
  transmitting on a legitimate user's pilot to corrupt the base
  station's channel estimate in its favour. This dissertation is about
  (b). Worth a sentence making that distinction explicit, it's an easy
  place for a reader unfamiliar with the area to get confused.
- Why (b) matters: once the eavesdropper has contaminated the estimate,
  the base station's subsequent beamforming can be steered toward
  leaking information to the attacker rather than (or in addition to)
  the legitimate user, this is the mechanism, not just "the attacker
  listens in."
- Physical-layer security as a field: detecting/preventing this at the
  signal level, as a complement to (not replacement for) upper-layer
  cryptography. One or two sentences positioning why physical-layer
  defences matter even when encryption exists (e.g. metadata/presence
  leakage, or contexts where key exchange itself is what's under
  attack).

## 1.2 Motivation

Points to cover:

- Why detection specifically (this work's task) rather than prevention
  or mitigation: detection is the necessary first step, and it's the
  formulation Hoang et al. (2021) [1], the source paper this work
  builds from, use, so it's also what makes direct comparison possible.
- Why machine learning for this task, as opposed to a fixed statistical
  threshold test: briefly foreshadow that a naive threshold is brittle
  to the fact that the detection statistic's distribution shifts with
  eavesdropper SNR (Chapter 5/6 show this quantitatively), motivating a
  learned decision boundary.
- Why *sequence-based* deep learning specifically, this is the
  dissertation's actual contribution beyond replicating Hoang et al.:
  the source paper (and most of the physical-layer detection literature,
  see Chapter 2) reduces the received-power trace to one or two summary
  statistics (mean, ratio) before classifying. That's a design choice,
  not a necessity, and it discards information. This work asks whether
  classifying the raw sequence directly, with models built for
  sequences (MLP on the full vector, LSTM, and a sequence autoencoder),
  recovers that discarded information as improved detection
  performance, and if so, how much of it, and under what conditions.

## 1.3 Aims and Objectives

**Aim.** To design, implement, and rigorously evaluate a sequence-based
deep learning approach for detecting active eavesdropping (pilot
contamination) attacks in a multi-user massive MIMO uplink, benchmarked
against classical machine learning baselines drawn from the physical-layer
security literature.

**Objectives.**

1. Reproduce a classical detection baseline (TC-SVM, Random Forest,
   XGBoost) from the physical-layer security literature [1] on simulated
   pilot-contamination data, and validate it against the source paper's
   published results.
2. Design and implement deep learning models, an MLP, an LSTM, and an
   autoencoder-based anomaly detector, that consume the full received-power
   sequence rather than hand-engineered summary statistics, and compare
   their detection performance against the classical baseline.
3. Establish a statistically rigorous evaluation protocol (cross-validation,
   significance testing, calibration analysis) sufficient to distinguish
   genuine performance differences between models from noise.
4. Evaluate the robustness of the best-performing detectors under
   conditions more representative of real-world deployment: severe class
   imbalance, varying eavesdropper signal strength, and adversarial
   evasion attempts.
5. Investigate, through ablation and interpretability analysis, whether the
   deep learning models' advantage (if any) arises from access to raw
   sequence information or from architectural capacity, and identify which
   parts of the input sequence drive their decisions.
6. Critically assess the practical deployment trade-offs (inference
   latency, model size, single-class vs. supervised training requirements)
   between the classical and deep learning approaches.

*This list is a proposal built from what the project actually became, not*
*copied from a proposal document, since none was supplied. Read it*
*critically: does it match what you'd have said your goals were at the*
*start? Edit freely. Whatever it says when you're done is what Chapter 7*
*needs to honestly assess yourself against, including admitting where an*
*objective was only partially met (both sample dissertations do this*
*openly, e.g. Sample 8 states plainly that ROUGE/BLEU evaluation was*
*never completed, that kind of honesty is expected, not penalised).*

## 1.4 Novelty and Contributions

*This section is drafted in full, since it's a factual claim about what*
*this work adds beyond the source paper, grounded in Chapters 3-6, not a*
*personal reflection. Read it critically against Chapter 6's honest*
*qualifications (§ 6.1 in particular tempers some of these) before*
*keeping it as-is: a novelty claim that ignores your own chapter 6*
*findings will read as inconsistent to an examiner who reads the whole*
*document.*

Hoang et al. (2021) [1] establish the system model, the MEAN/RATIO
feature construction, and the TC-SVM/SC-SVM baseline this dissertation
builds from. Relative to that starting point, this work's contributions
are:

1. **Sequence-consuming detection.** Where the source paper, and the
   related physical-layer authentication literature reviewed in Chapter
   2, reduce the received-power trace to one or two hand-engineered
   statistics before classifying, this work classifies the raw
   `T=50`-step sequence directly, via an MLP, an LSTM, and a sequence
   autoencoder, and measures what that costs or gains relative to the
   feature-engineered baseline (Chapter 5/6).
2. **A controlled representation ablation.** Rather than simply
   comparing a sequence model against a feature-based model (which
   confounds representation with architecture), § 6.2 holds model
   capacity fixed and varies only the amount of sequence information
   available, isolating how much of any performance difference is
   architecture and how much is information content. This experiment
   design, not just its result, is a contribution: it directly answers a
   question the headline comparison alone cannot.
3. **A statistically rigorous evaluation protocol.** Cross-validation,
   Wilcoxon/McNemar/Mann-Whitney significance testing with an explicit,
   stated distinction between paired and unpaired evidence strength (§
   3.4, § 6.1), and calibration/latency/model-size benchmarking, applied
   uniformly across all six models. The source paper reports point
   estimates; this work reports whether the differences between those
   estimates are distinguishable from noise.
4. **Convergent multi-method interpretability.** Four independent
   interpretability methods across two architecturally unrelated models
   (LSTM occlusion/permutation/Integrated Gradients, and autoencoder
   per-timestep reconstruction error) converge on the same window of the
   input sequence (§ 6.6), evidence not available from any single method
   alone.
5. **Physical-layer robustness beyond the source paper's SNR sweep.**
   An independent replication of Hoang et al.'s SNR-sensitivity curve (§
   6.8) extended to all six models (not only TC-SVM), plus a diagnosed
   (not merely observed) overfitting failure mode in the MLP at low SNR
   and a regularisation ablation testing candidate fixes (§ 6.5), neither
   of which has an analogue in the source paper.
6. **Adversarial and deployment-cost evaluation**, absent from the
   source paper entirely: feature-space evasion robustness (§ 6.7) and a
   systematic latency/model-size/parameter-count comparison across all
   six models (§ 6.4), directly relevant to whether any of these
   detectors could actually be deployed at a base station in real time.

## 1.5 Structure of the Report

Standard paragraph-per-chapter summary. Once Chapters 2-8 are finalised,
write one or two sentences per chapter here, e.g.:

> Chapter 2 reviews the physical-layer security literature this work sits
> within... Chapter 3 describes the system model and the six detection
> methods evaluated... Chapter 4 details the implementation, from the
> MATLAB channel simulator through to trained models... Chapter 5 presents
> results... Chapter 6 critically examines those results... Chapter 7
> reviews this work against the objectives stated above... Chapter 8
> concludes and outlines future work.
