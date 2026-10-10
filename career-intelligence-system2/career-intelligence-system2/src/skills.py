"""Shared helpers for turning free-text key-skills into countable "skill terms".

The Naukri "Keyskills" field is a bag of words/phrases with no delimiters
(e.g. ``"core java j2ee hibernate sql server"``), so we use 1-word and 2-word
terms (unigrams + bigrams) as the unit of a "skill".

Two different filters are used on purpose:
* the ML models see *all* terms except plain English stop-words
  (a word like "voice" or "fresher" is genuinely predictive), while
* analytics / skill-gap screens hide generic non-skill words
  (``GENERIC_TERMS``) so the output reads like real skills.
"""
from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# Keeps "c++", "c#", ".net", "node.js", "asp.net" as single tokens.
TOKEN_PATTERN = r"(?u)(?<![\w+#.])[a-z0-9.][a-z0-9+#.]*[a-z0-9+#]|\b[a-z0-9]\b"

# Words that appear in many postings but do not describe a learnable skill.
GENERIC_TERMS = frozenset("""
skills skill experience experienced fresher freshers executive executives manager
managers senior junior lead activities tools good excellent team handling office
center centre centres services service ms etc other various related working work
knowledge strong ability required candidate candidates job jobs profile role
position positions hiring opening openings walkin walk immediate joiner joiners
trainee trainees associate associates officer officers staff person persons level
new key core international domestic india indian company companies organization
developer developers engineer engineers analyst analysts specialist consultant professional
professionals head
""".split())


def split_terms(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, text.lower()) if isinstance(text, str) else []


def is_display_term(term: str) -> bool:
    """True if a unigram/bigram looks like a real skill (no generic words)."""
    parts = term.split()
    if not parts:
        return False
    if any(p in GENERIC_TERMS or p in ENGLISH_STOP_WORDS for p in parts):
        return False
    if all(p.isdigit() for p in parts):
        return False
    if len(term) < 2:
        return False
    return True


def make_vectorizer(min_df: int | float = 15, max_features: int | None = None) -> CountVectorizer:
    """Binary unigram+bigram skill vectoriser (fit it on TRAINING data only)."""
    return CountVectorizer(
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        stop_words=list(ENGLISH_STOP_WORDS),
        binary=True,
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
    )


def normalise_user_skills(skills: str | Iterable[str] | None) -> str:
    """Turn 'Python, SQL; machine learning' or a list into the dataset format."""
    if skills is None:
        return ""
    if not isinstance(skills, str):
        skills = " ".join(str(s) for s in skills)
    skills = skills.lower()
    skills = re.sub(r"[^a-z0-9+#.\s]", " ", skills)
    tokens = [t.rstrip(".") for t in skills.split()]
    return " ".join(t for t in tokens if re.search(r"[a-z0-9]", t))


# --------------------------------------------------------------------------- #
# SkillIndex: demand statistics for skill terms (overall and per role)
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd


class SkillIndex:
    """Counts how often each skill term appears, overall and inside each role.

    Fit it on whichever postings you are allowed to look at (all data for
    descriptive analytics, TRAINING data only when it feeds an evaluation).
    """

    def __init__(self, min_df: int = 10):
        self.min_df = min_df

    def fit(self, texts: pd.Series, labels: pd.Series) -> "SkillIndex":
        texts = texts.reset_index(drop=True)
        self.labels_ = labels.reset_index(drop=True).astype(str).to_numpy()
        self.vectorizer_ = make_vectorizer(min_df=self.min_df)
        self.X_ = self.vectorizer_.fit_transform(texts).tocsr()
        self.vocab_ = np.asarray(self.vectorizer_.get_feature_names_out())
        self.display_mask_ = np.array([is_display_term(t) for t in self.vocab_])
        self.n_docs_ = self.X_.shape[0]
        self.overall_count_ = np.asarray(self.X_.sum(axis=0)).ravel()
        return self

    # ---- overall -------------------------------------------------------- #
    def top_overall(self, n: int = 25) -> pd.DataFrame:
        df = pd.DataFrame({"skill": self.vocab_, "postings": self.overall_count_})
        df = df[self.display_mask_]
        df["share_pct"] = 100 * df["postings"] / self.n_docs_
        return df.sort_values("postings", ascending=False).head(n).reset_index(drop=True)

    # ---- per role ------------------------------------------------------- #
    def role_profile(self, role: str, min_support: int = 5, top: int = 40,
                     min_share: float = 0.03, lift_cut: float = 1.3) -> pd.DataFrame:
        """Skills most requested for `role`, with share, lift and a core/common tag."""
        mask = self.labels_ == role
        n_role = int(mask.sum())
        if n_role == 0:
            return pd.DataFrame(columns=["skill", "role_postings", "role_share",
                                         "overall_share", "lift", "kind"])
        role_count = np.asarray(self.X_[mask].sum(axis=0)).ravel()
        role_share = role_count / n_role
        overall_share = self.overall_count_ / self.n_docs_
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_share > 0, role_share / overall_share, 0.0)
        df = pd.DataFrame({
            "skill": self.vocab_, "role_postings": role_count,
            "role_share": role_share, "overall_share": overall_share, "lift": lift,
        })
        keep = self.display_mask_ & (role_count >= min_support) & (role_share >= min_share)
        df = df[keep].copy()
        df["kind"] = np.where(df["lift"] >= lift_cut, "core", "common")
        return df.sort_values("role_share", ascending=False).head(top).reset_index(drop=True)
