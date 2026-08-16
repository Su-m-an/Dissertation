# Abstract

Massive MIMO uplink systems rely on orthogonal pilot sequences to estimate
the channel state of each legitimate user. An active eavesdropper can
subvert this by transmitting on the same pilot as a target user during the
training phase, a pilot contamination attack that corrupts the base
station's channel estimate and can subsequently be exploited to intercept
the target's data. This dissertation designs, implements, and evaluates
machine learning detectors for this attack in a simulated `K=4` user,
`L=10` antenna uplink, following the detection framework of Hoang et al.
(2021) [1].

Two families of detector are compared. The first, following the source
literature, extracts two summary statistics (mean and ratio) from the
received power over a `T=50` sample training window and classifies with a
TC-SVM, Random Forest, or XGBoost. The second, the main contribution of
this work, instead consumes the full `T=50`-step power sequence directly,
using an MLP, an LSTM, and an autoencoder-based anomaly detector trained
only on legitimate traffic. Across a stratified 5-fold cross-validation
protocol with a documented hyperparameter search, the sequence-based LSTM
achieves the highest accuracy (99.13% ± 0.23%), a statistically significant
improvement over the classical baseline (Mann-Whitney U, p < 0.02) and a
modest, Wilcoxon-non-significant improvement over the MLP.

Beyond this headline comparison, the detectors are stress-tested under
severe class imbalance (down to a 999:1 legitimate-to-attack ratio), a
sweep of the eavesdropper's signal-to-noise ratio, and black-box
feature-space evasion attempts; interpreted using SHAP, timestep occlusion,
and per-timestep reconstruction error; and benchmarked for inference
latency and model size. A representation ablation isolates that the
sequence models' advantage stems primarily from access to the raw sequence
rather than architectural capacity alone, and a regularisation ablation
diagnoses, and partially addresses, an overfitting failure mode found in
the MLP at low eavesdropper SNR. The results position sequence-based deep
learning as a practical improvement over the classical baseline for this
task, while surfacing specific conditions, low SNR and severe class
imbalance chief among them, under which that advantage narrows or reverses.

*(~290 words. Check against your department's abstract word limit, both*
*samples this structure is based on kept theirs to roughly one page.)*
