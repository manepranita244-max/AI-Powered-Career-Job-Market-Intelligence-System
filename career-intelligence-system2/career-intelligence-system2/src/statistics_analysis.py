"""Module 2 - Statistical analysis of the job market.

Contents
--------
* descriptive statistics (mean, median, mode, variance, std, quartiles, IQR,
  skewness, kurtosis, coefficient of variation)
* correlation (Pearson and Spearman) with p-values
* outlier analysis (IQR rule and z-score rule)
* normality checks (decide between parametric / non-parametric tests)
* hypothesis tests with effect sizes and a multiple-testing correction
* experience-adjusted "skill salary premium" table

Every sentence of interpretation is produced from the computed numbers; nothing
is typed in by hand.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .eda import add_experience_band
from .preprocessing import load_clean
from .skills import SkillIndex

ALPHA = 0.05
MIN_N = config.MIN_SALARY_SUPPORT
RNG_SEED = config.RANDOM_STATE


# --------------------------------------------------------------------------- #
# 1. Descriptive statistics
# --------------------------------------------------------------------------- #

def descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        mode = s.mode()
        q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
        rows.append({
            "variable": c, "count": len(s), "mean": s.mean(), "median": q2,
            "mode": mode.iloc[0] if len(mode) else np.nan,
            "variance": s.var(ddof=1), "std_dev": s.std(ddof=1),
            "min": s.min(), "Q1": q1, "Q3": q3, "IQR": q3 - q1, "max": s.max(),
            "skewness": stats.skew(s), "kurtosis_excess": stats.kurtosis(s),
            "coef_variation_pct": 100 * s.std(ddof=1) / s.mean() if s.mean() else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Outliers
# --------------------------------------------------------------------------- #

def outlier_summary(s: pd.Series, name: str) -> dict:
    s = s.dropna().astype(float)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - s.mean()) / s.std(ddof=1)
    return {
        "variable": name, "n": len(s), "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_fence": lo, "upper_fence": hi,
        "iqr_outliers_low": int((s < lo).sum()), "iqr_outliers_high": int((s > hi).sum()),
        "iqr_outlier_pct": 100 * ((s < lo) | (s > hi)).mean(),
        "zscore_outliers_abs_gt_3": int((z.abs() > 3).sum()),
        "max_value": s.max(),
    }


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    sal = df["salary_mid_lpa"].dropna()
    items = [
        outlier_summary(sal, "salary_mid_lpa"),
        outlier_summary(np.log10(sal), "log10(salary_mid_lpa)"),
        outlier_summary(df["exp_mid"], "exp_mid"),
        outlier_summary(df["n_skill_words"][df["n_skill_words"] > 0], "n_skill_words"),
    ]
    return pd.DataFrame(items)


# --------------------------------------------------------------------------- #
# 3. Correlation
# --------------------------------------------------------------------------- #
CORR_COLS = ["exp_min", "exp_max", "exp_mid", "salary_mid_lpa", "n_skill_words", "n_cities"]


def correlation_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df[CORR_COLS].copy()
    d.loc[d["n_skill_words"] == 0, "n_skill_words"] = np.nan   # 0 means "not listed"
    return {"pearson": d.corr(method="pearson"), "spearman": d.corr(method="spearman")}


def correlation_tests(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.loc[d["n_skill_words"] == 0, "n_skill_words"] = np.nan
    d["log_salary"] = np.log(d["salary_mid_lpa"])
    pairs = [("exp_mid", "salary_mid_lpa"), ("exp_mid", "log_salary"),
             ("n_skill_words", "salary_mid_lpa"), ("n_cities", "salary_mid_lpa"),
             ("exp_mid", "n_skill_words")]
    rows = []
    for a, b in pairs:
        x = d[[a, b]].dropna()
        r, p = stats.pearsonr(x[a], x[b])
        rho, p_s = stats.spearmanr(x[a], x[b])
        rows.append({"x": a, "y": b, "n": len(x), "pearson_r": r, "pearson_p": p,
                     "spearman_rho": rho, "spearman_p": p_s})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Normality
# --------------------------------------------------------------------------- #

def normality_table(df: pd.DataFrame) -> pd.DataFrame:
    sal = df["salary_mid_lpa"].dropna()
    series = {"salary_mid_lpa": sal, "log(salary_mid_lpa)": np.log(sal),
              "exp_mid": df["exp_mid"].dropna()}
    rows = []
    for name, s in series.items():
        sample = s.sample(min(len(s), 5000), random_state=RNG_SEED)
        w, p_sw = stats.shapiro(sample)
        k2, p_k2 = stats.normaltest(s)
        rows.append({"variable": name, "n": len(s), "shapiro_W": w, "shapiro_p": p_sw,
                     "dagostino_K2": k2, "dagostino_p": p_k2,
                     "looks_normal_at_5pct": bool(p_sw > ALPHA and p_k2 > ALPHA)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Hypothesis tests
# --------------------------------------------------------------------------- #

def _bootstrap_median_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000):
    rng = np.random.default_rng(RNG_SEED)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def _size_label(value: float, kind: str) -> str:
    v = abs(value)
    cuts = {"cliffs_delta": (0.147, 0.33, 0.474), "cramers_v": (0.1, 0.3, 0.5),
            "epsilon_sq": (0.01, 0.06, 0.14), "spearman_rho": (0.1, 0.3, 0.5),
            "cohens_h": (0.2, 0.5, 0.8)}[kind]
    return ("negligible" if v < cuts[0] else "small" if v < cuts[1]
            else "medium" if v < cuts[2] else "large")


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    tests: list[dict] = []
    sal_df = df.dropna(subset=["salary_mid_lpa"]).copy()

    # --- H1: salary differs between role categories (Kruskal-Wallis) -------- #
    role_n = sal_df["role_category"].value_counts()
    roles = role_n[role_n >= MIN_N].index.tolist()
    g = [sal_df.loc[sal_df.role_category == r, "salary_mid_lpa"].to_numpy() for r in roles]
    H, p = stats.kruskal(*g)
    n = sum(len(x) for x in g)
    eps = (H - len(g) + 1) / (n - len(g))
    meds = sal_df[sal_df.role_category.isin(roles)].groupby("role_category")["salary_mid_lpa"].median()
    tests.append({
        "id": "H1", "question": "Do salaries differ across job role categories?",
        "H0": "All role categories have the same salary distribution",
        "test": f"Kruskal-Wallis ({len(roles)} roles with >= {MIN_N} salaried postings)",
        "statistic": H, "p_value": p, "n": n,
        "effect_name": "epsilon_sq", "effect_size": eps,
        "detail": f"highest median: {meds.idxmax()} ({meds.max():.2f} lakh); "
                  f"lowest median: {meds.idxmin()} ({meds.min():.2f} lakh)",
    })

    # --- H2: IT-software vs other industries (Mann-Whitney) ----------------- #
    it = sal_df["industry"].str.contains("IT-Software", na=False)
    a, b = sal_df.loc[it, "salary_mid_lpa"].to_numpy(), sal_df.loc[~it, "salary_mid_lpa"].to_numpy()
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = 2 * U / (len(a) * len(b)) - 1
    lo, hi = _bootstrap_median_diff(a, b)
    tests.append({
        "id": "H2", "question": "Do IT-Software postings pay differently from other industries?",
        "H0": "Salary distributions of IT-Software and other industries are identical",
        "test": "Mann-Whitney U (two-sided)", "statistic": U, "p_value": p, "n": len(a) + len(b),
        "effect_name": "cliffs_delta", "effect_size": delta,
        "detail": f"median IT-Software {np.median(a):.2f} vs others {np.median(b):.2f} lakh; "
                  f"bootstrap 95% CI of median difference [{lo:.2f}, {hi:.2f}]",
    })

    # --- H3: experience vs salary (Spearman) -------------------------------- #
    x = sal_df.dropna(subset=["exp_mid"])
    rho, p = stats.spearmanr(x["exp_mid"], x["salary_mid_lpa"])
    tests.append({
        "id": "H3", "question": "Is higher required experience associated with higher salary?",
        "H0": "No monotonic association between experience and salary (rho = 0)",
        "test": "Spearman rank correlation", "statistic": rho, "p_value": p, "n": len(x),
        "effect_name": "spearman_rho", "effect_size": rho,
        "detail": "rank-based, so robust to the skewed salary distribution",
    })

    # --- H4: salary differs between cities (Kruskal-Wallis) ----------------- #
    city_n = sal_df["primary_city"].value_counts()
    cities = city_n[city_n >= MIN_N].index.tolist()
    g = [sal_df.loc[sal_df.primary_city == c, "salary_mid_lpa"].to_numpy() for c in cities]
    H, p = stats.kruskal(*g)
    n = sum(len(x) for x in g)
    eps = (H - len(g) + 1) / (n - len(g))
    meds = sal_df[sal_df.primary_city.isin(cities)].groupby("primary_city")["salary_mid_lpa"].median()
    tests.append({
        "id": "H4", "question": "Do salaries differ between the main cities?",
        "H0": "All cities have the same salary distribution",
        "test": f"Kruskal-Wallis ({', '.join(cities)})",
        "statistic": H, "p_value": p, "n": n,
        "effect_name": "epsilon_sq", "effect_size": eps,
        "detail": "medians (lakh): " + ", ".join(f"{c} {meds[c]:.2f}" for c in cities)
                  + ". Cities differ in role/experience mix, so this is not a pure city effect.",
    })

    # --- H5: role mix depends on city (chi-square) -------------------------- #
    top_roles = df["role_category"].value_counts().head(8).index
    top_cities = df["primary_city"].value_counts().head(4).index
    d = df[df["role_category"].isin(top_roles) & df["primary_city"].isin(top_cities)]
    ct = pd.crosstab(d["role_category"], d["primary_city"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
    tests.append({
        "id": "H5", "question": "Does the mix of job roles depend on the city?",
        "H0": "Role category and city are independent",
        "test": "Chi-square test of independence (top 8 roles x top 4 cities)",
        "statistic": chi2, "p_value": p, "n": int(ct.to_numpy().sum()),
        "effect_name": "cramers_v", "effect_size": v,
        "detail": f"dof={dof}; smallest expected cell count = {expected.min():.1f}"
                  + (" (assumption OK)" if expected.min() >= 5 else " (<5: interpret with care)"),
    })

    # --- H6: entry-level jobs and pay transparency (two proportions) -------- #
    d = df.dropna(subset=["exp_min"]).copy()
    fresher = d["exp_min"] == 0
    ct = pd.crosstab(fresher, d["salary_disclosed"])
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    p1 = d.loc[fresher, "salary_disclosed"].mean()
    p2 = d.loc[~fresher, "salary_disclosed"].mean()
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    tests.append({
        "id": "H6", "question": "Are fresher-friendly postings more (or less) likely to disclose pay? (SDG 8 transparency)",
        "H0": "Salary-disclosure rate is the same for fresher-open and experience-only postings",
        "test": "Chi-square test of two proportions", "statistic": chi2, "p_value": p, "n": len(d),
        "effect_name": "cohens_h", "effect_size": h,
        "detail": f"disclosure rate: fresher-open {100*p1:.1f}% vs experience-required {100*p2:.1f}%",
    })

    # --- H7: BPO/voice vs programming experience requirement ---------------- #
    a = df.loc[df.role_category == "Programming & Design", "exp_mid"].dropna().to_numpy()
    b = df.loc[df.role_category == "Voice", "exp_mid"].dropna().to_numpy()
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = 2 * U / (len(a) * len(b)) - 1
    tests.append({
        "id": "H7", "question": "Do Programming & Design roles ask for more experience than Voice (BPO) roles?",
        "H0": "Experience requirement distributions are identical",
        "test": "Mann-Whitney U (two-sided)", "statistic": U, "p_value": p, "n": len(a) + len(b),
        "effect_name": "cliffs_delta", "effect_size": delta,
        "detail": f"median required experience: Programming {np.median(a):.1f} vs Voice {np.median(b):.1f} years",
    })

    out = pd.DataFrame(tests)
    out["p_holm"] = _holm(out["p_value"].tolist())
    out["reject_H0_raw"] = out["p_value"] < ALPHA
    out["reject_H0_holm"] = out["p_holm"] < ALPHA
    out["effect_label"] = [_size_label(v, k) for v, k in zip(out["effect_size"], out["effect_name"])]
    out["interpretation"] = out.apply(_interpret, axis=1)
    return out


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _interpret(r: pd.Series) -> str:
    decision = ("Reject H0" if r["reject_H0_holm"] else
                "Reject H0 only before correction" if r["reject_H0_raw"] else "Fail to reject H0")
    return (f"{decision} at alpha={ALPHA} ({_fmt_p(r['p_value'])}, Holm-adjusted "
            f"{_fmt_p(r['p_holm'])}); effect size {r['effect_name']} = {r['effect_size']:.3f} "
            f"({r['effect_label']}). {str(r['detail']).rstrip('.')}.")


# --------------------------------------------------------------------------- #
# 6. Experience-adjusted skill salary premium
# --------------------------------------------------------------------------- #

def skill_salary_premium(df: pd.DataFrame, min_with: int = MIN_N, top_terms: int = 150) -> pd.DataFrame:
    """For common skills: salary of postings that ask for the skill vs those that do not.

    Salary rises strongly with experience, so we first remove the experience
    effect (quadratic fit of log-salary on experience) and compare residuals.
    This is descriptive evidence, not a causal claim.
    """
    d = df[(df["skills_text"] != "") & df["salary_mid_lpa"].notna() & df["exp_mid"].notna()].reset_index(drop=True)
    if len(d) < 100:
        return pd.DataFrame()
    y = np.log(d["salary_mid_lpa"].to_numpy())
    X = np.column_stack([np.ones(len(d)), d["exp_mid"], d["exp_mid"] ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    idx = SkillIndex(min_df=min_with).fit(d["skills_text"], pd.Series(["all"] * len(d)))
    counts = idx.overall_count_
    order = np.argsort(-counts)
    rows = []
    for j in order:
        term = idx.vocab_[j]
        if not idx.display_mask_[j]:
            continue
        has = idx.X_[:, j].toarray().ravel() > 0
        if has.sum() < min_with or (~has).sum() < min_with:
            continue
        u, p = stats.mannwhitneyu(resid[has], resid[~has], alternative="two-sided")
        rows.append({
            "skill": term, "salaried_postings_with_skill": int(has.sum()),
            "median_salary_with": float(np.median(d.loc[has, "salary_mid_lpa"])),
            "median_salary_without": float(np.median(d.loc[~has, "salary_mid_lpa"])),
            "exp_adjusted_premium_pct": 100 * (np.exp(np.median(resid[has]) - np.median(resid[~has])) - 1),
            "p_value": p,
        })
        if len(rows) >= top_terms:
            break
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_fdr"] = stats.false_discovery_control(out["p_value"].to_numpy(), method="bh")
    out["significant_fdr_5pct"] = out["q_value_fdr"] < ALPHA
    return out.sort_values("exp_adjusted_premium_pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_statistics(df: pd.DataFrame | None = None) -> dict:
    df = load_clean() if df is None else df
    df = df.copy()
    # 0 means "not listed" for these two counts, so treat it as missing
    df["n_skill_words"] = df["n_skill_words"].where(df["n_skill_words"] > 0)
    df["n_cities"] = df["n_cities"].where(df["n_cities"] > 0)
    desc = descriptive_stats(df, ["exp_min", "exp_max", "exp_mid", "salary_mid_lpa", "n_skill_words", "n_cities"])
    desc_by_band = (add_experience_band(df).dropna(subset=["salary_mid_lpa", "exp_band"])
                    .groupby("exp_band", observed=True)["salary_mid_lpa"]
                    .describe(percentiles=[.25, .5, .75]).reset_index())
    corr = correlation_matrices(df)
    out = {
        "descriptive": desc, "salary_by_experience_band": desc_by_band,
        "outliers": outlier_table(df), "normality": normality_table(df),
        "correlation_pearson": corr["pearson"].reset_index(),
        "correlation_spearman": corr["spearman"].reset_index(),
        "correlation_tests": correlation_tests(df),
        "hypothesis_tests": hypothesis_tests(df),
        "skill_salary_premium": skill_salary_premium(df),
    }
    for name, t in out.items():
        t.to_csv(config.TABLE_DIR / f"stats_{name}.csv", index=False)
    with open(config.TABLE_DIR / "stats_hypothesis_tests.json", "w") as f:
        json.dump(out["hypothesis_tests"].to_dict(orient="records"), f, indent=2, default=float)
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    res = run_statistics()
    print(res["descriptive"].round(3).to_string(), "\n")
    print(res["outliers"].round(2).to_string(), "\n")
    print(res["normality"].round(4).to_string(), "\n")
    print(res["correlation_tests"].round(4).to_string(), "\n")
    for _, r in res["hypothesis_tests"].iterrows():
        print(f"{r['id']}: {r['question']}\n   {r['interpretation']}\n")
    p = res["skill_salary_premium"]
    print(p.head(8).round(3).to_string()); print(p.tail(5).round(3).to_string())
