"""Module 3c - Salary prediction (only exposed in the app if it passes the checks).

Honest-modelling choices
------------------------
* Only ~29% of postings disclose pay, and the target is the *midpoint of the
  advertised range*, not what people are finally paid.  The model therefore
  predicts "typical advertised salary", never an individual's actual salary.
* Target = log(salary in INR lakh p.a.) because salary is strongly right-skewed.
* Same leakage rules as the role model: pipeline-internal preprocessing,
  company-grouped split and grouped cross-validation.
* Two baselines are reported so you can see what the ML adds:
  (a) global median, (b) experience-only quadratic regression.
* The prediction interval is built from out-of-fold residuals on the training
  set and its real coverage is then measured on the untouched test set.
* A reliability verdict is computed automatically from thresholds in config.py.

Run:  python -m src.salary_model
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from . import config
from .features import FEATURE_COLUMNS, add_group_column, build_preprocessor
from .preprocessing import load_clean, sha256_of

SEED = config.RANDOM_STATE
SALARY_FEATURES = FEATURE_COLUMNS + ["role_category"]


def prepare_salary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["salary_mid_lpa"].notna() & (df["skills_text"] != "")].copy()
    d["role_category"] = d["role_category"].fillna("Unknown")
    d = add_group_column(d)
    d["log_salary"] = np.log(d["salary_mid_lpa"])
    return d


# --------------------------------------------------------------------------- #
def make_salary_pipeline(name: str, p: dict) -> Pipeline:
    prep = build_preprocessor(skill_min_df=p.get("skill_min_df", 5), extra_categorical=("role_category",))
    if name == "Ridge":
        reg = Ridge(alpha=p["alpha"])
    elif name == "Random Forest":
        reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=p["min_samples_leaf"],
                                    max_features=p["max_features"], n_jobs=-1, random_state=SEED)
    elif name == "Gradient Boosting":
        reg = GradientBoostingRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                        max_depth=p["max_depth"], subsample=0.8,
                                        max_features=p["max_features"], random_state=SEED)
    else:
        raise ValueError(name)
    return Pipeline([("prep", prep), ("reg", reg)])


def salary_grid() -> dict[str, list[dict]]:
    return {
        "Ridge": [{"skill_min_df": m, "alpha": a} for m in (3, 5) for a in (1, 3, 10, 30, 100, 300)],
        "Random Forest": [{"skill_min_df": 5, "min_samples_leaf": l, "max_features": f}
                          for l in (2, 5) for f in (0.3, 0.6)],
        "Gradient Boosting": [{"skill_min_df": 5, "n_estimators": n, "learning_rate": 0.05,
                               "max_depth": dpt, "max_features": 0.3}
                              for n in (200, 400) for dpt in (3, 4)],
    }


def regression_metrics(y_true_lpa: np.ndarray, pred_lpa: np.ndarray) -> dict:
    y, p = np.asarray(y_true_lpa, float), np.asarray(pred_lpa, float)
    ly, lp = np.log(y), np.log(p)
    ss_res, ss_tot = ((ly - lp) ** 2).sum(), ((ly - ly.mean()) ** 2).sum()
    ss_res_o, ss_tot_o = ((y - p) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    ape = np.abs(y - p) / y
    return {
        "r2_log": 1 - ss_res / ss_tot, "rmse_log": float(np.sqrt(ss_res / len(y))),
        "mae_lakh": float(np.mean(np.abs(y - p))), "median_abs_error_lakh": float(np.median(np.abs(y - p))),
        "rmse_lakh": float(np.sqrt(np.mean((y - p) ** 2))), "r2_lakh": 1 - ss_res_o / ss_tot_o,
        "median_abs_pct_error": float(100 * np.median(ape)),
        "within_30pct_share": float(100 * np.mean(ape <= 0.30)),
    }


class ExperienceOnlyBaseline:
    """log(salary) ~ a + b*exp + c*exp^2, fitted on the training rows."""

    def fit(self, exp, y_lpa):
        exp = np.nan_to_num(np.asarray(exp, float), nan=np.nanmedian(exp))
        X = np.column_stack([np.ones(len(exp)), exp, exp ** 2])
        self.beta_, *_ = np.linalg.lstsq(X, np.log(y_lpa), rcond=None)
        self.fill_ = float(np.median(exp))
        return self

    def predict(self, exp):
        exp = np.nan_to_num(np.asarray(exp, float), nan=self.fill_)
        return np.exp(np.column_stack([np.ones(len(exp)), exp, exp ** 2]) @ self.beta_)


def main(log=print) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    d = prepare_salary_dataset(load_clean())
    tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                                        random_state=SEED).split(d, groups=d["group"]))
    train, test = d.iloc[tr_i].reset_index(drop=True), d.iloc[te_i].reset_index(drop=True)
    assert not (set(train["group"]) & set(test["group"])), "company leakage!"
    log(f"salaried rows: train={len(train)} test={len(test)}  (company overlap = 0)")
    Xtr, ytr, gtr = train[SALARY_FEATURES], train["log_salary"], train["group"]
    Xte = test[SALARY_FEATURES]
    y_te_lpa = test["salary_mid_lpa"].to_numpy()

    # ---- cross-validated tuning (train only) ------------------------------ #
    folds = list(GroupKFold(n_splits=config.CV_FOLDS).split(Xtr, ytr, groups=gtr))
    cv_rows = []
    for name, configs in salary_grid().items():
        for p in configs:
            ts = time.time()
            rm, r2 = [], []
            for a, b in folds:
                m = make_salary_pipeline(name, p).fit(Xtr.iloc[a], ytr.iloc[a])
                pr = m.predict(Xtr.iloc[b])
                rm.append(np.sqrt(np.mean((pr - ytr.iloc[b]) ** 2)))
                r2.append(1 - ((pr - ytr.iloc[b]) ** 2).sum() / ((ytr.iloc[b] - ytr.iloc[b].mean()) ** 2).sum())
            cv_rows.append({"model": name, "params": json.dumps(p), "cv_rmse_log": np.mean(rm),
                            "cv_rmse_log_std": np.std(rm), "cv_r2_log": np.mean(r2), "seconds": time.time() - ts})
            log(f"  {name:18s} {p}  CV RMSE(log)={np.mean(rm):.3f}  R2={np.mean(r2):.3f}  ({time.time()-ts:.0f}s)")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(config.TABLE_DIR / "salary_cv_results.csv", index=False)

    # ---- test evaluation (once) ------------------------------------------- #
    rows, fitted = [], {}
    med = float(np.median(train["salary_mid_lpa"]))
    rows.append({"model": "Baseline: global median", "params": "{}",
                 **regression_metrics(y_te_lpa, np.full(len(test), med))})
    base = ExperienceOnlyBaseline().fit(train["exp_mid"], train["salary_mid_lpa"])
    rows.append({"model": "Baseline: experience only (quadratic)", "params": "{}",
                 **regression_metrics(y_te_lpa, base.predict(test["exp_mid"]))})
    best_by_model = {}
    for name, g in cv.groupby("model"):
        top = g.sort_values("cv_rmse_log").iloc[0]
        p = json.loads(top["params"])
        best_by_model[name] = p
        m = make_salary_pipeline(name, p).fit(Xtr, ytr)
        fitted[name] = m
        rows.append({"model": name, "params": top["params"], "cv_rmse_log": top["cv_rmse_log"],
                     "cv_r2_log": top["cv_r2_log"],
                     **regression_metrics(y_te_lpa, np.exp(m.predict(Xte)))})
    test_df = pd.DataFrame(rows)
    test_df.to_csv(config.TABLE_DIR / "salary_test_metrics.csv", index=False)
    models_only = test_df[test_df["model"].isin(best_by_model)]
    best_name = models_only.sort_values("cv_rmse_log").iloc[0]["model"]
    best_params = best_by_model[best_name]
    log(f"\nBest salary model (lowest CV RMSE on log scale): {best_name} {best_params}")
    log(test_df.round(3).drop(columns=["params"]).to_string(index=False))

    # ---- prediction interval from out-of-fold residuals -------------------- #
    oof = np.zeros(len(train))
    for a, b in folds:
        oof[b] = make_salary_pipeline(best_name, best_params).fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b])
    resid = ytr.to_numpy() - oof
    q_lo, q_hi = np.percentile(resid, [10, 90])
    pred_te = fitted[best_name].predict(Xte)
    coverage = float(np.mean((test["log_salary"] >= pred_te + q_lo) & (test["log_salary"] <= pred_te + q_hi)))
    log(f"80% interval multipliers: x{np.exp(q_lo):.2f} .. x{np.exp(q_hi):.2f}; empirical test coverage = {coverage:.1%}")

    # ---- reliability verdict ------------------------------------------------ #
    best_row = test_df[test_df.model == best_name].iloc[0]
    gm = test_df[test_df.model == "Baseline: global median"].iloc[0]
    eo = test_df[test_df.model == "Baseline: experience only (quadratic)"].iloc[0]
    gain_vs_median = 100 * (gm["mae_lakh"] - best_row["mae_lakh"]) / gm["mae_lakh"]
    gain_vs_exp = 100 * (eo["mae_lakh"] - best_row["mae_lakh"]) / eo["mae_lakh"]
    checks = {
        f"train rows >= {config.SALARY_MIN_TRAIN_ROWS}": bool(len(train) >= config.SALARY_MIN_TRAIN_ROWS),
        f"test R2 (log) >= {config.SALARY_MIN_R2_LOG}": bool(best_row["r2_log"] >= config.SALARY_MIN_R2_LOG),
        f"MAE >= {config.SALARY_MIN_MAE_GAIN_PCT:.0f}% better than global median":
            bool(gain_vs_median >= config.SALARY_MIN_MAE_GAIN_PCT),
    }
    reliable = all(checks.values())
    log(f"\nReliability checks: {checks}  ->  reliable = {reliable}")
    log(f"MAE gain vs global median: {gain_vs_median:.1f}%  |  vs experience-only: {gain_vs_exp:.1f}%")

    role_support = train.groupby("role_category").size().to_dict()
    role_support_all = d.groupby("role_category").size().to_dict()

    # ---- final fit on ALL salaried rows & save ------------------------------ #
    final = make_salary_pipeline(best_name, best_params).fit(d[SALARY_FEATURES], d["log_salary"])
    joblib.dump(final, config.MODELS_DIR / "salary_model.joblib", compress=3)
    meta = {
        "model_name": best_name, "params": best_params, "reliable": reliable, "checks": checks,
        "interval_log_low": float(q_lo), "interval_log_high": float(q_hi),
        "interval_nominal": 0.80, "interval_test_coverage": coverage,
        "test_metrics": {k: float(v) for k, v in best_row.items() if isinstance(v, (int, float, np.floating))},
        "mae_gain_vs_global_median_pct": float(gain_vs_median),
        "mae_gain_vs_experience_only_pct": float(gain_vs_exp),
        "n_train": int(len(train)), "n_test": int(len(test)), "n_final_fit": int(len(d)),
        "role_salary_support": {str(k): int(v) for k, v in role_support_all.items()},
        "features": SALARY_FEATURES, "unit": "INR lakh per annum (advertised range midpoint)",
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "data_sha256": sha256_of(config.RAW_PATH),
        "caveats": [
            "Only ~29% of postings disclose pay; disclosure is not random.",
            "Target is the midpoint of the advertised range, not realised pay.",
            "Postings are from 2015-2016; amounts are historical, not today's market.",
        ],
    }
    with open(config.MODELS_DIR / "salary_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame({"actual_lpa": y_te_lpa, "predicted_lpa": np.exp(pred_te),
                  "low_80": np.exp(pred_te + q_lo), "high_80": np.exp(pred_te + q_hi),
                  "role_category": test["role_category"]}).to_csv(
        config.TABLE_DIR / "salary_test_predictions.csv", index=False)
    log(f"Done in {time.time()-t0:.0f}s")
    return meta


if __name__ == "__main__":
    main()
