"""
statistical_tests.py

Consumes the outputs of cv_tuning_classical.py and cv_tuning_neural.py and
runs the documented significance tests.

Important methodological note, made explicit rather than glossed over:
TC-SVM/Random Forest/XGBoost are trained and evaluated on Data/ATD.csv
(184,000 rows, MEAN/RATIO features). MLP/LSTM/Autoencoder are trained and
evaluated on Data/ATD_sequence.csv (4,000 rows, E1-E50 features) -- a
different, smaller sample set. That means the two groups' CV folds and
test-set rows are NOT the same underlying samples, so a *paired* test
(Wilcoxon on fold pairs, McNemar's on matched predictions) is only valid
WITHIN each group:

  - Wilcoxon signed-rank + McNemar's: valid within {TC-SVM, RF, XGBoost}
    and within {MLP, LSTM, Autoencoder}, since each group shares identical
    fold splits and test rows.
  - Across groups (e.g. "is LSTM significantly better than XGBoost" - the
    actual headline claim of the whole comparison): NOT a valid paired
    test. Reported instead with the weaker, unpaired Mann-Whitney U test
    on fold-level accuracy distributions, explicitly labelled as such.

Wilcoxon signed-rank is used over a paired t-test because a paired t-test
assumes the per-fold differences are approximately normal, an assumption
that is difficult to justify with only k=5 folds; Wilcoxon does not
require it.
"""

import json
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, binomtest

OUT = "experiments/phase1_statistical_rigor"

CLASSICAL = ["tc_svm", "random_forest", "xgboost"]
NEURAL = ["mlp", "lstm", "autoencoder"]
ALL_MODELS = CLASSICAL + NEURAL


def load_cv_accuracy(tag):
    df = pd.read_csv(f"{OUT}/results/{tag}_cv_folds.csv")
    return df.sort_values("fold")["accuracy"].values


def load_predictions(tag):
    return pd.read_csv(f"{OUT}/results/{tag}_test_predictions.csv")


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    # discordant pairs: b = A right/B wrong, c = A wrong/B right
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "note": "no discordant pairs"}
    # exact binomial form of McNemar's test (appropriate given b+c well
    # above the small-sample threshold here, and avoids a chi-square
    # continuity-correction judgment call)
    result = binomtest(min(b, c), n, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "p_value": float(result.pvalue)}


def wilcoxon_test(acc_a, acc_b):
    diffs = acc_a - acc_b
    if np.allclose(diffs, 0):
        return {"statistic": 0.0, "p_value": 1.0, "note": "identical fold accuracies"}
    stat, p = wilcoxon(acc_a, acc_b)
    return {"statistic": float(stat), "p_value": float(p)}


def mannwhitney_test(acc_a, acc_b):
    stat, p = mannwhitneyu(acc_a, acc_b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


if __name__ == "__main__":

    cv_acc = {m: load_cv_accuracy(m) for m in ALL_MODELS}
    preds = {m: load_predictions(m) for m in ALL_MODELS}

    summary = {"cv_accuracy_mean_std": {}, "within_group_paired_tests": [], "cross_group_unpaired_tests": []}

    for m in ALL_MODELS:
        summary["cv_accuracy_mean_std"][m] = {
            "mean": float(np.mean(cv_acc[m])), "std": float(np.std(cv_acc[m]))
        }

    print("=== CV accuracy summary ===")
    for m in ALL_MODELS:
        s = summary["cv_accuracy_mean_std"][m]
        print(f"  {m:15s} {s['mean']:.4f} +/- {s['std']:.4f}")

    print("\n=== Within-group paired tests (Wilcoxon + McNemar) ===")
    for group_name, group in [("classical", CLASSICAL), ("neural", NEURAL)]:
        for a, b in itertools.combinations(group, 2):
            wt = wilcoxon_test(cv_acc[a], cv_acc[b])

            df_a, df_b = preds[a], preds[b]
            assert np.array_equal(df_a["y_true"].values, df_b["y_true"].values), \
                f"{a} and {b} test sets do not match -- cannot pair"
            mt = mcnemar_test(df_a["y_true"].values, df_a["y_pred"].values, df_b["y_pred"].values)

            entry = {"group": group_name, "model_a": a, "model_b": b,
                      "wilcoxon": wt, "mcnemar": mt}
            summary["within_group_paired_tests"].append(entry)
            print(f"  [{group_name}] {a} vs {b}: Wilcoxon p={wt['p_value']:.4f}  "
                  f"McNemar p={mt['p_value']:.4f} (b={mt['b']}, c={mt['c']})")

    print("\n=== Cross-group unpaired tests (Mann-Whitney U) -- weaker evidence, see docstring ===")
    for a, b in itertools.product(CLASSICAL, NEURAL):
        ut = mannwhitney_test(cv_acc[a], cv_acc[b])
        summary["cross_group_unpaired_tests"].append({"model_a": a, "model_b": b, "mannwhitney": ut})
        print(f"  {a} vs {b}: Mann-Whitney U p={ut['p_value']:.4f}")

    with open(f"{OUT}/results/statistical_tests_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUT}/results/statistical_tests_summary.json")
