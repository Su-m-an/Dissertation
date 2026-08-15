"""
cv_tuning_classical.py

Phase 1 statistical rigor for the classical models (TC-SVM, Random Forest,
XGBoost), trained on the MEAN/RATIO features from Data/ATD.csv.

For each model:
  1. Documented hyperparameter search (grid documented in *_search_grid.json).
     TC-SVM is searched on a stratified 15,000-row subsample (3-fold CV) --
     RBF-kernel SVM training cost scales roughly O(n^2.2-2.5), and a full
     grid at the full 184,000-row scale was estimated (from a pilot run) at
     3-7 hours for the search alone. RF/XGBoost are searched at full scale
     since tree ensembles are cheap here.
  2. The winning hyperparameters are then evaluated with real stratified
     5-fold CV *at full scale* for all three models -- this is the number
     that gets reported, not the subsample search itself.
  3. A final model is refit on the full 80% training split (same split as
     the corrected baseline, random_state=42) and saved, together with its
     predictions on the held-out 20% test set, for the downstream
     statistical-testing and calibration/latency stages.

Writes only under experiments/phase1_statistical_rigor/ -- results/ (the
corrected baseline) is never touched.
"""

import json
import time
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

SEED = 42
K = 5
OUT = "experiments/phase1_statistical_rigor"

os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/models", exist_ok=True)

data = pd.read_csv("Data/ATD.csv")
X_all = data[["MEAN", "RATIO"]].values
y_all = data["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)


def fold_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
    }


def cv_score_config(model_factory, params, X, y, k=K, seed=SEED):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        model = model_factory(**params)
        model.fit(X_tr, y[train_idx])
        pred = model.predict(X_val)
        accs.append(accuracy_score(y[val_idx], pred))
    return float(np.mean(accs))


def full_cv_report(model_factory, params, X, y, tag, k=K, seed=SEED):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    fold_num = 0
    for train_idx, val_idx in skf.split(X, y):
        fold_num += 1
        assert len(set(train_idx) & set(val_idx)) == 0, "leakage: index overlap"

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = model_factory(**params)
        model.fit(X_tr, y[train_idx])
        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]

        m = fold_metrics(y[val_idx], pred, prob)
        m["fold"] = fold_num
        rows.append(m)
        print(f"    [{tag} fold {fold_num}/{k}] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

    return pd.DataFrame(rows)


def run_model(tag, model_factory, grid, search_on_subsample=False, subsample_n=15000, search_k=3):
    print(f"\n=== {tag} ===")
    t0 = time.time()

    if search_on_subsample:
        _, X_sub, _, y_sub = train_test_split(
            X_train, y_train, test_size=subsample_n, random_state=SEED, stratify=y_train
        )
        search_X, search_y, used_k = X_sub, y_sub, search_k
        print(f"  Hyperparameter search on stratified subsample (n={subsample_n}, k={search_k})")
    else:
        search_X, search_y, used_k = X_train, y_train, K
        print(f"  Hyperparameter search at full scale (n={len(X_train)}, k={used_k})")

    search_results = []
    best_params, best_score = None, -np.inf
    for params in ParameterGrid(grid):
        score = cv_score_config(model_factory, params, search_X, search_y, k=used_k)
        search_results.append({**params, "cv_accuracy": score})
        print(f"    params={params} -> cv_acc={score:.4f}")
        if score > best_score:
            best_score, best_params = score, params

    search_time = time.time() - t0
    print(f"  Search complete in {search_time:.1f}s. Best: {best_params} (cv_acc={best_score:.4f})")

    with open(f"{OUT}/results/{tag}_search_grid.json", "w") as f:
        json.dump({"grid": grid, "results": search_results, "search_time_s": search_time,
                    "searched_on_subsample": search_on_subsample, "search_k": used_k}, f, indent=2)
    with open(f"{OUT}/results/{tag}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    t1 = time.time()
    print(f"  Full-scale {K}-fold CV with best params...")
    cv_df = full_cv_report(model_factory, best_params, X_train, y_train, tag)
    cv_time = time.time() - t1
    cv_df.to_csv(f"{OUT}/results/{tag}_cv_folds.csv", index=False)

    print(f"  CV accuracy: {cv_df['accuracy'].mean():.4f} +/- {cv_df['accuracy'].std():.4f}  "
          f"(full-scale CV time: {cv_time:.1f}s)")

    t2 = time.time()
    scaler = StandardScaler()
    X_tr_full = scaler.fit_transform(X_train)
    X_te_full = scaler.transform(X_test)
    final_model = model_factory(**best_params)
    final_model.fit(X_tr_full, y_train)
    refit_time = time.time() - t2

    pred = final_model.predict(X_te_full)
    prob = final_model.predict_proba(X_te_full)[:, 1]

    joblib.dump(final_model, f"{OUT}/models/{tag}_tuned.joblib")
    joblib.dump(scaler, f"{OUT}/models/{tag}_tuned_scaler.joblib")

    pd.DataFrame({"y_true": y_test, "y_pred": pred, "y_prob": prob}).to_csv(
        f"{OUT}/results/{tag}_test_predictions.csv", index=False
    )

    total_time = time.time() - t0
    timing = {
        "search_time_s": search_time, "cv_time_s": cv_time, "final_refit_time_s": refit_time,
        "total_time_s": total_time
    }
    with open(f"{OUT}/results/{tag}_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print(f"  {tag} done in {total_time:.1f}s total.")
    return cv_df, timing


if __name__ == "__main__":

    overall_start = time.time()

    run_model(
        "tc_svm",
        lambda **p: SVC(kernel="rbf", probability=True, random_state=SEED, **p),
        grid={"C": [0.1, 1, 10], "gamma": [0.0001, 0.001, 0.01]},
        search_on_subsample=True, subsample_n=15000, search_k=3
    )

    run_model(
        "random_forest",
        lambda **p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p),
        grid={"n_estimators": [100, 200, 300], "max_depth": [10, 15, 20]},
        search_on_subsample=False
    )

    run_model(
        "xgboost",
        lambda **p: XGBClassifier(random_state=SEED, eval_metric="logloss", **p),
        grid={"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        search_on_subsample=False
    )

    print(f"\nAll classical models complete in {time.time() - overall_start:.1f}s total.")
