"""Module 1b - Exploratory Data Analysis (EDA).

Every function takes the clean dataframe and returns a small pandas table, so
the same numbers can feed the notebook-style report AND the Streamlit dashboard.
`run_eda()` saves all tables (CSV) and static figures (PNG) under ``reports/``.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .preprocessing import load_clean
from .skills import SkillIndex

EXP_BINS = [-0.01, 1, 3, 5, 8, 12, 60]
EXP_LABELS = ["0-1 yrs", "1-3 yrs", "3-5 yrs", "5-8 yrs", "8-12 yrs", "12+ yrs"]
MIN_N = config.MIN_SALARY_SUPPORT

PALETTE = {"main": "#1f6f8b", "accent": "#e07a1f", "grey": "#8a8f98", "green": "#2e8b57"}


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def add_experience_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exp_band"] = pd.cut(df["exp_mid"], bins=EXP_BINS, labels=EXP_LABELS)
    return df


def overview_kpis(df: pd.DataFrame) -> dict:
    """Headline numbers (all computed from the data)."""
    sal = df["salary_mid_lpa"].dropna()
    city_share = df["primary_city"].value_counts(normalize=True)
    ind_share = df["industry"].value_counts(normalize=True)
    return {
        "total_postings": int(len(df)),
        "unique_companies": int(df["company"].nunique()),
        "role_categories": int(df["role_category"].nunique()),
        "industries": int(df["industry"].nunique()),
        "cities": int(df["primary_city"].nunique()),
        "median_exp_min": float(df["exp_min"].median()),
        "fresher_open_pct": float(100 * (df["exp_min"] == 0).sum() / df["exp_min"].notna().sum()),
        "salary_disclosed_pct": float(100 * df["salary_disclosed"].mean()),
        "median_salary_lpa": float(sal.median()) if len(sal) else float("nan"),
        "top3_city_share_pct": float(100 * city_share.head(3).sum()),
        "top_industry": str(ind_share.index[0]),
        "top_industry_share_pct": float(100 * ind_share.iloc[0]),
    }


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["role_category", "skills_text", "exp_mid", "ug_degree", "pg_degree",
            "primary_city", "industry", "salary_mid_lpa"]
    out = []
    for c in cols:
        s = df[c]
        missing = s.isna() | (s == "") if s.dtype == object or str(s.dtype) == "str" else s.isna()
        out.append({"column": c, "missing": int(missing.sum()),
                    "missing_pct": round(100 * missing.mean(), 2)})
    return pd.DataFrame(out)


def _salary_stats(g: pd.DataFrame) -> pd.Series:
    sal = g["salary_mid_lpa"].dropna()
    return pd.Series({
        "postings": len(g),
        "salaried_postings": len(sal),
        "median_exp_mid": g["exp_mid"].median(),
        "median_salary_lpa": sal.median() if len(sal) >= MIN_N else np.nan,
    })


def group_summary(df: pd.DataFrame, col: str, top: int = 15) -> pd.DataFrame:
    """Postings, share, typical experience and median salary per category."""
    keep = df[col].value_counts().head(top).index
    out = df[df[col].isin(keep)].groupby(col).apply(_salary_stats, include_groups=False)
    out["share_pct"] = 100 * out["postings"] / len(df)
    out = out.sort_values("postings", ascending=False).reset_index()
    out["postings"] = out["postings"].astype(int)
    out["salaried_postings"] = out["salaried_postings"].astype(int)
    return out


def experience_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = add_experience_band(df).dropna(subset=["exp_mid"])
    g = d.groupby("exp_band", observed=True)
    out = g.apply(_salary_stats, include_groups=False).reset_index()
    out["share_pct"] = 100 * out["postings"] / len(d)
    return out


def top_skills(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    d = df[(df["skills_text"] != "")]
    return SkillIndex().fit(d["skills_text"], d["role_category"].fillna("Unknown")).top_overall(n)


def skills_by_role(df: pd.DataFrame, role: str, n: int = 15) -> pd.DataFrame:
    d = df[(df["skills_text"] != "")]
    idx = SkillIndex().fit(d["skills_text"], d["role_category"].fillna("Unknown"))
    return idx.role_profile(role, top=n)


def role_city_matrix(df: pd.DataFrame, top_roles: int = 8, top_cities: int = 5) -> pd.DataFrame:
    roles = df["role_category"].value_counts().head(top_roles).index
    cities = df["primary_city"].value_counts().head(top_cities).index
    d = df[df["role_category"].isin(roles) & df["primary_city"].isin(cities)]
    ct = pd.crosstab(d["role_category"], d["primary_city"], normalize="index") * 100
    return ct.round(1)


# --------------------------------------------------------------------------- #
# Static figures (Matplotlib) - the dashboard uses Plotly versions of the same
# --------------------------------------------------------------------------- #

def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _barh(ax, labels, values, color, fmt="{:,.0f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([str(l)[:38] for l in labels])
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8)


def save_figures(df: pd.DataFrame) -> list[str]:
    saved = []

    def _save(fig, name):
        path = config.FIG_DIR / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path.name)

    roles = group_summary(df, "role_category", 15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _barh(ax, roles["role_category"], roles["postings"], PALETTE["main"])
    _style(ax, "Top 15 job role categories (number of postings)")
    _save(fig, "01_top_roles")

    ind = group_summary(df, "industry", 12)
    fig, ax = plt.subplots(figsize=(8, 5))
    _barh(ax, ind["industry"], ind["postings"], PALETTE["accent"])
    _style(ax, "Top 12 industries (number of postings)")
    _save(fig, "02_top_industries")

    cities = group_summary(df, "primary_city", 8)
    fig, ax = plt.subplots(figsize=(7, 4))
    _barh(ax, cities["primary_city"], cities["postings"], PALETTE["green"])
    _style(ax, "Postings by primary city")
    _save(fig, "03_cities")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["exp_min"].dropna(), bins=range(0, 21), color=PALETTE["main"], edgecolor="white")
    _style(axes[0], "Minimum experience asked", "years", "postings")
    es = experience_summary(df)
    axes[1].bar(es["exp_band"].astype(str), es["share_pct"], color=PALETTE["accent"])
    _style(axes[1], "Share of postings by experience band (midpoint)", "", "% of postings")
    _save(fig, "04_experience")

    sal = df["salary_mid_lpa"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(sal, bins=40, color=PALETTE["main"], edgecolor="white")
    axes[0].axvline(sal.median(), color=PALETTE["accent"], ls="--", label=f"median = {sal.median():.2f}")
    axes[0].legend()
    _style(axes[0], f"Salary midpoint (n={len(sal)} disclosed)", "INR lakh per annum", "postings")
    axes[1].hist(np.log10(sal), bins=40, color=PALETTE["green"], edgecolor="white")
    _style(axes[1], "Same, on log10 scale (more symmetric)", "log10(INR lakh)", "postings")
    _save(fig, "05_salary_distribution")

    d = add_experience_band(df).dropna(subset=["salary_mid_lpa", "exp_band"])
    groups = [g["salary_mid_lpa"].to_numpy() for _, g in d.groupby("exp_band", observed=True)]
    labels = [str(k) for k, _ in d.groupby("exp_band", observed=True)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#cfe4ec"), medianprops=dict(color=PALETTE["accent"]))
    _style(ax, "Salary by experience band (outliers hidden)", "", "INR lakh per annum")
    _save(fig, "06_salary_by_experience")

    ts = top_skills(df, 20)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _barh(ax, ts["skill"], ts["share_pct"], PALETTE["main"], fmt="{:.1f}%")
    _style(ax, "Most requested skill terms (% of postings with skills listed)")
    _save(fig, "07_top_skills")

    rs = group_summary(df, "role_category", 25).dropna(subset=["median_salary_lpa"])
    rs = rs.sort_values("median_salary_lpa", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    _barh(ax, [f"{r} (n={n})" for r, n in zip(rs["role_category"], rs["salaried_postings"])],
          rs["median_salary_lpa"], PALETTE["accent"], fmt="{:.2f}")
    _style(ax, f"Median salary by role (only roles with >= {MIN_N} salaried postings)",
           "INR lakh per annum")
    _save(fig, "08_salary_by_role")
    return saved


def run_eda(df: pd.DataFrame | None = None) -> dict:
    df = load_clean() if df is None else df
    tables = {
        "eda_role_summary": group_summary(df, "role_category", 25),
        "eda_industry_summary": group_summary(df, "industry", 20),
        "eda_city_summary": group_summary(df, "primary_city", 12),
        "eda_experience_summary": experience_summary(df),
        "eda_top_skills": top_skills(df, 40),
        "eda_missingness": missingness(df),
        "eda_role_city_matrix_pct": role_city_matrix(df).reset_index(),
    }
    for name, t in tables.items():
        t.to_csv(config.TABLE_DIR / f"{name}.csv", index=False)
    figs = save_figures(df)
    kpis = overview_kpis(df)
    with open(config.TABLE_DIR / "overview_kpis.json", "w") as f:
        json.dump(kpis, f, indent=2)
    return {"kpis": kpis, "tables": list(tables), "figures": figs}


if __name__ == "__main__":
    out = run_eda()
    print(json.dumps(out["kpis"], indent=2))
    print("tables:", out["tables"])
    print("figures:", out["figures"])
