"""Module 3a - Build the supervised-learning table and the preprocessing pipeline.

LEAKAGE RULES followed in this project
--------------------------------------
1. The target is ``role_category``.  The columns ``role``, ``functional_area``,
   ``job_title`` and the raw description are *never* used as features because
   they either are the label's sibling taxonomy or simply spell the answer out.
2. All learned preprocessing (skill vocabulary, one-hot categories, medians,
   scaling) lives inside a scikit-learn ``Pipeline`` so it is fitted on the
   training rows only.
3. Train/test split is grouped by company: postings from the same employer are
   often near-copies, so they must not be spread across both sides.
4. Model selection uses cross-validation inside the training part; the test part
   is evaluated once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .skills import make_vectorizer

# What a job seeker can tell us -> which dataset column it maps to
FEATURE_COLUMNS = ["skills_text", "exp_mid", "ug_degree", "pg_degree", "cities", "industry"]
TARGET = "role_category"
FEATURE_GROUPS = {
    "skills": ["skills_text"],
    "experience": ["exp_mid"],
    "education": ["ug_degree", "pg_degree"],
    "location": ["cities"],
    "interest (industry)": ["industry"],
}


def pipe_tokenizer(text: str) -> list[str]:
    """'Bengaluru|Chennai' -> ['Bengaluru', 'Chennai'] (module-level => picklable)."""
    return [t for t in text.split("|") if t]


def add_group_column(d: pd.DataFrame) -> pd.DataFrame:
    """Add a 'group' column = company (anonymous employers each get their own group)."""
    d = d.reset_index(drop=True).copy()
    anon = d["company"].str.lower().isin(["confidential", "unknown", "", "not disclosed"])
    d["group"] = np.where(anon, "anon_" + d.index.astype(str), d["company"].str.lower())
    return d


def prepare_role_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for role prediction + only role families with enough examples."""
    d = df[df[TARGET].notna() & (df["skills_text"] != "")].copy()
    d = d[~d[TARGET].isin(config.EXCLUDED_ROLE_LABELS)]
    counts = d[TARGET].value_counts()
    keep = counts[counts >= config.MIN_ROLE_SAMPLES].index
    d = d[d[TARGET].isin(keep)]
    return add_group_column(d)


def grouped_stratified_split(d: pd.DataFrame, test_fold: int = 0):
    """80/20 split: stratified by role, grouped by company. Returns index arrays."""
    sgkf = StratifiedGroupKFold(n_splits=int(round(1 / config.TEST_SIZE)), shuffle=True,
                                random_state=config.RANDOM_STATE)
    splits = list(sgkf.split(d, d[TARGET], groups=d["group"]))
    train_idx, test_idx = splits[test_fold]
    return train_idx, test_idx


def build_preprocessor(skill_min_df: int = 15, city_min_df: int = 20,
                       drop_groups: tuple[str, ...] = (),
                       extra_categorical: tuple[str, ...] = ()) -> ColumnTransformer:
    """Everything that learns from data is inside here (=> fitted on train only)."""
    blocks = []
    if "skills" not in drop_groups:
        blocks.append(("skills", make_vectorizer(min_df=skill_min_df), "skills_text"))
    if "location" not in drop_groups:
        blocks.append(("cities", CountVectorizer(tokenizer=pipe_tokenizer, token_pattern=None,
                                                 lowercase=False, binary=True, min_df=city_min_df),
                       "cities"))
    cat_cols = [c for g, cols in (("education", ["ug_degree", "pg_degree"]),
                                  ("interest (industry)", ["industry"]))
                if g not in drop_groups for c in cols] + list(extra_categorical)
    if cat_cols:
        blocks.append(("categories", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                     sparse_output=True)),
        ]), cat_cols))
    if "experience" not in drop_groups:
        blocks.append(("experience", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]), ["exp_mid"]))
    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=1.0)


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name == "skills":
            names += [f"skill: {t}" for t in trans.get_feature_names_out()]
        elif name == "cities":
            names += [f"city: {t}" for t in trans.get_feature_names_out()]
        elif name == "categories":
            names += [str(t) for t in trans.named_steps["onehot"].get_feature_names_out(cols)]
        else:
            names += list(cols)
    return names
