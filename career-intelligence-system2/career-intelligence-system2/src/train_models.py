"""Module 3b - Train, tune, compare and save the job-role prediction models.

Protocol (no data leakage)
--------------------------
1. Split once into TRAIN (80% of companies) and TEST (20% of companies).
2. Tune every model with 5-fold *grouped, stratified* cross-validation on TRAIN.
   Preprocessing is inside the pipeline, so each fold learns its own vocabulary.
3. Pick the best setting per model by cross-validated macro-F1.
4. Refit on TRAIN, evaluate on TEST exactly once -> the numbers shown in the app.
5. Report uncertainty with a company-level bootstrap on the test predictions.
6. Refit the winning setting on ALL labelled rows and save it for the app.

Run:  python -m src.train_models
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from . import config
from .features import (FEATURE_COLUMNS, FEATURE_GROUPS, TARGET, build_preprocessor,
                       grouped_stratified_split, prepare_role_dataset)
from .preprocessing import load_clean, sha256_of

SEED = config.RANDOM_STATE


# --------------------------------------------------------------------------- #
# Candidate models and their (small) tuning grids
# --------------------------------------------------------------------------- #

def make_estimator(name: str, p: dict):
    if name == "Logistic Regression":
        return LogisticRegression(C=p["C"], max_iter=1000)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                      random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=p["min_samples_leaf"],
                                      max_features="sqrt", n_jobs=-1, random_state=SEED)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                          learning_rate=0.1, subsample=0.8, max_features="sqrt",
                                          random_state=SEED)
    raise ValueError(name)


def candidate_grid() -> dict[str, list[dict]]:
    def expand(**axes):
        keys = list(axes)
        return [dict(zip(keys, vals)) for vals in itertools.product(*axes.values())]
    return {
        "Logistic Regression": expand(skill_min_df=[2, 3], balanced=[False, True], C=[0.3, 1.0]),
        "Decision Tree": expand(skill_min_df=[3], balanced=[False, True], max_depth=[10, 20, None],
                                min_samples_leaf=[2, 5]),
        "Random Forest": expand(skill_min_df=[3], balanced=[False, True], min_samples_leaf=[1, 2]),
        "Gradient Boosting": expand(skill_min_df=[3], balanced=[False, True], n_estimators=[100],
                                    max_depth=[3, 5]),
    }


def make_pipeline(name: str, p: dict, drop_groups: tuple[str, ...] = ()) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(skill_min_df=p.get("skill_min_df", 3), drop_groups=drop_groups)),
        ("clf", make_estimator(name, p)),
    ])


def fit_pipeline(pipe: Pipeline, X, y, balanced: bool) -> Pipeline:
    if balanced:
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def topk_accuracy(proba: np.ndarray, classes: np.ndarray, y_true, k: int) -> float:
    top = np.argsort(-proba, axis=1)[:, :k]
    hits = [(classes[row] == t).any() for row, t in zip(top, np.asarray(y_true))]
    return float(np.mean(hits))


def metric_row(y_true, y_pred, proba=None, classes=None) -> dict:
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
    }
    if proba is not None:
        row["top3_accuracy"] = topk_accuracy(proba, classes, y_true, 3)
        row["top5_accuracy"] = topk_accuracy(proba, classes, y_true, 5)
    return row


# --------------------------------------------------------------------------- #
# Cross-validation (training rows only)
# --------------------------------------------------------------------------- #

def cross_validate_grid(X: pd.DataFrame, y: pd.Series, groups: pd.Series, log=print) -> pd.DataFrame:
    folds = list(StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True,
                                      random_state=SEED).split(X, y, groups=groups))
    rows = []
    for name, configs in candidate_grid().items():
        for p in configs:
            t0 = time.time()
            f1s, accs = [], []
            for tr, va in folds:
                pipe = fit_pipeline(make_pipeline(name, p), X.iloc[tr], y.iloc[tr], p["balanced"])
                pred = pipe.predict(X.iloc[va])
                m = metric_row(y.iloc[va], pred)
                f1s.append(m["f1_macro"])
                accs.append(m["accuracy"])
            rows.append({"model": name, "params": json.dumps(p, default=str),
                         "cv_f1_macro_mean": np.mean(f1s), "cv_f1_macro_std": np.std(f1s),
                         "cv_accuracy_mean": np.mean(accs), "cv_accuracy_std": np.std(accs),
                         "seconds": time.time() - t0})
            log(f"  {name:20s} {p}  F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
                f"acc={np.mean(accs):.3f}  ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def best_per_model(cv: pd.DataFrame) -> dict[str, dict]:
    best = {}
    for name, g in cv.groupby("model"):
        top = g.sort_values(["cv_f1_macro_mean", "cv_accuracy_mean"], ascending=False).iloc[0]
        best[name] = json.loads(top["params"])
    return best


# --------------------------------------------------------------------------- #
# Bootstrap (resample companies, keep each company's rows together)
# --------------------------------------------------------------------------- #

def group_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                       n_boot: int = 300) -> dict:
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    accs, f1s = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        m = metric_row(y_true[idx], y_pred[idx])
        accs.append(m["accuracy"])
        f1s.append(m["f1_macro"])
    lo_a, hi_a = np.percentile(accs, [2.5, 97.5])
    lo_f, hi_f = np.percentile(f1s, [2.5, 97.5])
    return {"accuracy_ci_low": lo_a, "accuracy_ci_high": hi_a,
            "f1_macro_ci_low": lo_f, "f1_macro_ci_high": hi_f}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l[:22] for l in labels]
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted role")
    ax.set_ylabel("True role")
    ax.set_title(title + "  (cell = count, colour = row-normalised recall)", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_comparison(test_df: pd.DataFrame, path) -> None:
    d = test_df[~test_df["model"].str.startswith("Baseline")]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(d))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, d[m], w, label=m.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    ax.set_title("Held-out test performance (companies never seen in training)", loc="left", fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(log=print) -> dict:
    warnings.filterwarnings("ignore")
    t_start = time.time()
    df = load_clean()
    d = prepare_role_dataset(df)
    train_idx, test_idx = grouped_stratified_split(d)
    train, test = d.iloc[train_idx].reset_index(drop=True), d.iloc[test_idx].reset_index(drop=True)
    Xtr, ytr, gtr = train[FEATURE_COLUMNS], train[TARGET], train["group"]
    Xte, yte = test[FEATURE_COLUMNS], test[TARGET]
    overlap = set(train["group"]) & set(test["group"])
    assert not overlap, "company leakage between train and test!"
    log(f"rows: train={len(train)}  test={len(test)}  classes={d[TARGET].nunique()}  "
        f"company overlap={len(overlap)}")

    # ---- 1. cross-validated tuning on TRAIN only ------------------------- #
    log("\n[1/5] Cross-validated tuning on the training set ...")
    cv = cross_validate_grid(Xtr, ytr, gtr, log)
    cv.to_csv(config.TABLE_DIR / "ml_cv_results.csv", index=False)
    best = best_per_model(cv)

    # ---- 2. one-time test evaluation ------------------------------------- #
    log("\n[2/5] Refit best setting of each model on TRAIN, evaluate on TEST once ...")
    classes = None
    rows, per_class, cms, fitted, preds = [], {}, {}, {}, {}
    for strategy in ("most_frequent", "stratified"):
        dummy = DummyClassifier(strategy=strategy, random_state=SEED).fit(Xtr, ytr)
        pred = dummy.predict(Xte)
        proba = dummy.predict_proba(Xte)
        rows.append({"model": f"Baseline: {strategy.replace('_', ' ')}", "params": "{}",
                     **metric_row(yte, pred, proba, dummy.classes_)})
    for name, p in best.items():
        pipe = fit_pipeline(make_pipeline(name, p), Xtr, ytr, p["balanced"])
        pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)
        classes = pipe.named_steps["clf"].classes_
        ci = group_bootstrap_ci(yte.to_numpy(), pred, test["group"].to_numpy())
        cv_row = cv[(cv.model == name) & (cv.params == json.dumps(p, default=str))].iloc[0]
        rows.append({"model": name, "params": json.dumps(p, default=str),
                     "cv_f1_macro": cv_row["cv_f1_macro_mean"], "cv_accuracy": cv_row["cv_accuracy_mean"],
                     **metric_row(yte, pred, proba, classes), **ci})
        rep = precision_recall_fscore_support(yte, pred, labels=classes, zero_division=0)
        per_class[name] = pd.DataFrame({"role": classes, "precision": rep[0], "recall": rep[1],
                                        "f1": rep[2], "support": rep[3]})
        cms[name] = confusion_matrix(yte, pred, labels=classes)
        fitted[name], preds[name] = pipe, pred
        log(f"  {name:20s} acc={rows[-1]['accuracy']:.3f}  macroF1={rows[-1]['f1_macro']:.3f}  "
            f"top3={rows[-1]['top3_accuracy']:.3f}")
    test_df = pd.DataFrame(rows)
    test_df.to_csv(config.TABLE_DIR / "ml_test_metrics.csv", index=False)

    models_only = test_df[~test_df["model"].str.startswith("Baseline")]
    best_name = (models_only.sort_values(["cv_f1_macro", "cv_accuracy"], ascending=False)
                 .iloc[0]["model"])
    best_params = best[best_name]
    log(f"\nBest model by cross-validated macro-F1: {best_name} {best_params}")

    for name in cms:
        pd.DataFrame(cms[name], index=classes, columns=classes).to_csv(
            config.TABLE_DIR / f"ml_confusion_{name.replace(' ', '_').lower()}.csv")
        per_class[name].to_csv(config.TABLE_DIR / f"ml_per_class_{name.replace(' ', '_').lower()}.csv", index=False)
    plot_confusion(cms[best_name], list(classes), f"Confusion matrix - {best_name} (test set)",
                   config.FIG_DIR / "09_confusion_matrix_best.png")
    plot_model_comparison(test_df, config.FIG_DIR / "10_model_comparison.png")

    # ---- 3. feature-group ablation (descriptive, on TEST) ----------------- #
    log("\n[3/5] Feature-group ablation for the best model ...")
    abl = []
    full = test_df[test_df.model == best_name].iloc[0]
    abl.append({"removed_group": "(none - full model)", "accuracy": full["accuracy"], "f1_macro": full["f1_macro"]})
    for g in FEATURE_GROUPS:
        pipe = fit_pipeline(make_pipeline(best_name, best_params, drop_groups=(g,)), Xtr, ytr,
                            best_params["balanced"])
        m = metric_row(yte, pipe.predict(Xte))
        abl.append({"removed_group": g, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
    abl = pd.DataFrame(abl)
    abl["f1_macro_change"] = abl["f1_macro"] - full["f1_macro"]
    abl.to_csv(config.TABLE_DIR / "ml_ablation.csv", index=False)
    log(abl.round(3).to_string(index=False))

    # ---- 4. refit on ALL labelled rows and save --------------------------- #
    log("\n[4/5] Refit best setting on ALL labelled rows and save ...")
    final = fit_pipeline(make_pipeline(best_name, best_params), d[FEATURE_COLUMNS], d[TARGET],
                         best_params["balanced"])
    joblib.dump(final, config.MODELS_DIR / "role_model.joblib", compress=3)
    meta = {
        "model_name": best_name, "params": best_params,
        "classes": [str(c) for c in final.named_steps["clf"].classes_],
        "feature_columns": FEATURE_COLUMNS, "target": TARGET,
        "n_rows_train_eval": int(len(train)), "n_rows_test": int(len(test)),
        "n_rows_final_fit": int(len(d)),
        "test_metrics": {k: float(v) for k, v in full.items() if isinstance(v, (int, float, np.floating))},
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "data_sha256": sha256_of(config.RAW_PATH),
        "note": "Metrics are from the model fitted on TRAIN only; the saved model is refit on all rows.",
    }
    with open(config.MODELS_DIR / "role_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- 5. split definition (so other modules can reuse the same test set) #
    log("\n[5/5] Saving split definition ...")
    split = pd.DataFrame({"job_id": pd.concat([train["job_id"], test["job_id"]]),
                          "split": ["train"] * len(train) + ["test"] * len(test)})
    split.to_csv(config.TABLE_DIR / "ml_split.csv", index=False)
    log(f"Done in {time.time() - t_start:.0f}s")
    return {"best": best_name, "test": test_df}


if __name__ == "__main__":
    main()
