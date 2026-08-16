# 8. Conclusion

*Status: scaffold. Both samples keep this short (3-4 pages), a summary*
*and a future-work list, not new analysis. Draft it last, once Chapters*
*6-7 are finalised, so it accurately reflects what you actually concluded*
*there rather than what was planned at the outset.*

## 8.1 Summary

One or two paragraphs, points to hit:

- What the dissertation set out to do (§ 1.3's aim, one sentence).
- What was built: a validated reproduction of Hoang et al.'s classical
  baseline, and three sequence-consuming deep learning detectors compared
  against it under a statistically rigorous, multi-phase evaluation
  protocol (CV and significance testing, class-imbalance and SNR
  robustness, evasion testing, interpretability).
- The headline result, stated with the qualification § 6.1/7.2 earned:
  the LSTM achieves the best cross-validated accuracy (99.13%), but the
  strongest form of evidence for "sequence models beat classical models"
  is the weakest statistical test used in this dissertation, and the
  representation ablation shows the advantage is mostly about the full
  sequence recovering what a two-feature summary already captured, not
  architecture alone.
- One sentence on the autoencoder as a genuinely distinct practical
  option (single-class, smallest model, near-best PR-AUC), not simply a
  weaker classifier.

## 8.2 Key Findings, Restated Briefly

Bullet list, one line each, cross-referencing the chapter/section they
came from, e.g.:

- Statistically indistinguishable classical models (§ 6.1).
- `MEAN` alone nearly matches the full 50-step sequence (§ 6.2).
- Convergent interpretability evidence for timesteps 36-46 across four
  methods and two model types (§ 6.6).
- MLP overfits severely at low eavesdropper SNR; regularisation narrows
  but does not close the gap (§ 6.5).
- Deep learning models resist black-box feature-space evasion
  considerably better than classical models, with a named,
  unresolved dimensionality confound (§ 6.7).
- SNR-sensitivity curve independently replicates Hoang et al. (2021)
  within 1-2 points (§ 6.8).

## 8.3 Limitations

Short version of § 6.9, two or three sentences, this chapter isn't the
place to re-argue them in detail, just to acknowledge they stand.

## 8.4 Real-World Applications and Deployment Scenarios

*This section is drafted in full: it's domain reasoning about where this*
*kind of detector would plausibly be used, grounded in the deployment-cost*
*results already established in Chapter 5/6, not a personal reflection.*
*Trim or reweight it toward whatever application your supervisor or*
*programme finds most relevant.*

The detectors evaluated in this dissertation are trained on a simulated
`K=4, L=10` massive MIMO uplink, so any deployment claim here is
necessarily about *transferability of the approach*, not a demonstrated
result on the named system, that caveat from § 6.9 applies throughout
this section.

**5G/6G massive MIMO base stations.** The setting this dissertation
directly simulates. Pilot contamination is a known, actively-studied
vulnerability in massive MIMO specifically because pilot sequences are
reused across cells for scalability (§ 2.1). A detector integrated into
the base station's channel-estimation pipeline would need to run within
the coherence time of the channel, on the order of milliseconds; Table
5.5's latency figures (sub-millisecond for every model except TC-SVM)
suggest the MLP, LSTM, and autoencoder are compatible with this
constraint in principle, while TC-SVM's 0.377 ms/sample and comparatively
large model size make it the weakest of the six on this specific
criterion, consistent with § 6.4's finding.

**Private 5G and industrial IoT.** Factory, warehouse, and critical
infrastructure deployments increasingly use private 5G/massive-MIMO
cells with a small, known set of legitimate devices, closer to this
dissertation's `K=4` user setting than a public macro-cell with hundreds
of users. Pan et al. [5] (§ 2.3) validate ML-based physical-layer
detection on exactly this kind of industrial CSI data with field
hardware, precedent this dissertation's approach could plausibly follow
toward hardware validation, named as future work below.

**Contexts without labelled attack data.** Any deployment where an
operator cannot assume representative examples of the specific attack
they're defending against, a genuinely common constraint, since
adversaries adapt and historical attack examples may not represent a
novel one, is exactly the scenario the autoencoder (§ 6.3) is suited to:
trained entirely on normal traffic, no labelled attack examples required.
Worth naming explicitly as the practical argument for including a
single-class model at all, beyond matching Hoang et al.'s SC-SVM.

**Resource-constrained edge deployment.** Small-cell and edge base
stations, or any deployment with tight power/memory budgets, would
weight model size and latency (§ 6.4) more heavily than the headline
accuracy comparison in § 6.1. On that criterion, the autoencoder (23 KB)
and MLP (71 KB) are far more deployable than Random Forest (25 MB) or
TC-SVM (840 KB, growing with training-set size via its support vector
count), independent of their relative accuracy.

**Beyond cellular: UAV swarms, V2X, and satellite uplinks.** Any wireless
system relying on pilot-based or reference-signal-based channel
estimation among a known set of legitimate transmitters, drone swarm
coordination links, vehicle-to-everything (V2X) safety channels, or
satellite uplinks with a fixed set of ground stations, faces a
structurally similar attack surface. Applying this dissertation's
approach there would require re-deriving the system model (§ 3.1) for
the relevant channel statistics (e.g. line-of-sight-dominated Rician
fading for satellite links, rather than the Rayleigh fading assumed
here), not simply reusing the trained models as-is.

## 8.5 Future Work

Concrete, specific items, not a generic list, drawn directly from gaps
named in Chapters 2, 6, and 7:

- A paired cross-family significance test (evaluate classical models on
  the sequence dataset, or vice versa), to put § 6.1's central claim on
  firmer statistical ground.
- Ground the evasion-robustness perturbation in the physical simulator,
  translating a feature-space distance into an actually achievable
  `rho_E`/fading realisation, and resolve the dimensionality confound
  between the 2-feature and 50-feature evasion results (§ 6.7).
- Investigate the physical cause of the LSTM/autoencoder's convergent
  attention on timesteps 36-46 (§ 6.6), is this a property of the
  Rayleigh fading process, the `RATIO` statistic's accumulation
  behaviour, or something else.
- Extend the classical baseline to Hoang et al.'s full four-kernel SVM
  comparison and their `gamma`-overfitting study (§ 7.1).
- A deeper literature review pass to close the gap named in § 2.4
  (broader ML-for-wireless-security work beyond the five physical-layer
  security papers used here).
- Validation beyond the simulated `K=4, L=10` Rayleigh-fading model used
  throughout, real or hardware-in-the-loop data (e.g. USRP-based, following
  Pan et al.'s validation approach [5]), or a broader sweep of `K`/`L`,
  was out of scope here and would be the natural next step toward a
  deployable system.
- Extend the system model itself (§ 3.1) to fading distributions relevant
  to the non-cellular applications named in § 8.4, Rician fading for
  satellite/line-of-sight links in particular, to test whether the
  representation-ablation finding (§ 6.2, a two-statistic summary
  capturing most of a sequence model's advantage) still holds outside the
  Rayleigh-fading setting it was established in.
- A resource-constrained deployment study directly comparing the
  autoencoder and MLP (the two smallest, lowest-latency models, § 6.4) on
  actual edge hardware, rather than the model-size/latency proxies used
  here, to validate § 8.4's edge-deployment argument with real
  measurements.
