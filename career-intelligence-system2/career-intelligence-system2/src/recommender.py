"""Module 4 - Career recommendation engine.

Input : skills, education, experience, location, interests
Output: top roles, match score, reasons, skill gaps, learning roadmap

Match score (0-100) = weighted blend of five components, each between 0 and 1:

  ml          trained classifier's probability for the role      (weight 0.55)
  skills      share of the role's in-demand skills you already have (0.25)
  experience  is your experience inside the role's typical range?   (0.10)
  location    is the role as common in your city as overall?        (0.05)
  interest    is the role as common in your chosen industry?        (0.05)

The weights are a transparent design choice (change them in WEIGHTS).  They are
NOT tuned on the test set; `evaluate()` reports honestly whether the blend helps.
The score is a ranking index, not a probability of getting hired.
"""
from __future__ import annotations

import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, TARGET, prepare_role_dataset
from .preprocessing import load_clean
from .skills import SkillIndex, normalise_user_skills

WEIGHTS = {"ml": 0.55, "skills": 0.25, "experience": 0.10, "location": 0.05, "interest": 0.05}
CORE_TOP = 20


def dedupe_terms(terms: list[str]) -> list[str]:
    """Drop a one-word term if a longer term in the list already contains it."""
    out = []
    for t in terms:
        if " " not in t and any(t in o.split() for o in terms if o != t and " " in o):
            continue
        out.append(t)
    return out


class CareerRecommender:
    def __init__(self, d: pd.DataFrame, model, salary_model=None, salary_meta: dict | None = None,
                 weights: dict | None = None):
        """`d` = prepared role dataset (rows the recommender may learn statistics from)."""
        self.model = model
        self.classes = np.asarray(model.named_steps["clf"].classes_)
        self.weights = weights or WEIGHTS
        self.salary_model, self.salary_meta = salary_model, salary_meta or {}
        d = d[d[TARGET].isin(self.classes)].reset_index(drop=True)
        self.d = d
        self.skill_index = SkillIndex(min_df=5).fit(d["skills_text"], d[TARGET])
        self._analyzer = self.skill_index.vectorizer_.build_analyzer()
        self._vocab = set(self.skill_index.vocab_)
        self._build_role_stats()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_artifacts(cls) -> "CareerRecommender":
        d = prepare_role_dataset(load_clean())
        model = joblib.load(config.MODELS_DIR / "role_model.joblib")
        sm_path = config.MODELS_DIR / "salary_model.joblib"
        sal_model = joblib.load(sm_path) if sm_path.exists() else None
        meta_path = config.MODELS_DIR / "salary_model_metadata.json"
        sal_meta = json.load(open(meta_path)) if meta_path.exists() else {}
        return cls(d, model, sal_model, sal_meta)

    # ------------------------------------------------------------------ #
    def _build_role_stats(self) -> None:
        d = self.d
        city_lists = d["cities"].map(lambda s: [c for c in s.split("|") if c])
        self.n_docs = len(d)
        overall_city = Counter(c for cs in city_lists for c in set(cs))
        self.overall_city_share = {c: n / self.n_docs for c, n in overall_city.items()}
        self.overall_ind_share = (d["industry"].value_counts() / self.n_docs).to_dict()
        self.role_stats: dict[str, dict] = {}
        self.core_skills: dict[str, pd.DataFrame] = {}
        for role in self.classes:
            m = (d[TARGET] == role).to_numpy()
            g = d[m]
            cl = Counter(c for cs in city_lists[m] for c in set(cs))
            sal = g["salary_mid_lpa"].dropna()
            prof = self.skill_index.role_profile(role, top=40)
            core = prof[prof["kind"] == "core"]
            self.core_skills[role] = (core if len(core) >= 5 else prof).head(CORE_TOP).reset_index(drop=True)
            self.role_stats[role] = {
                "n_postings": int(m.sum()),
                "exp_q1": float(g["exp_mid"].quantile(0.25)), "exp_median": float(g["exp_mid"].median()),
                "exp_q3": float(g["exp_mid"].quantile(0.75)),
                "city_share": {c: n / m.sum() for c, n in cl.items()},
                "industry_share": (g["industry"].value_counts() / m.sum()).to_dict(),
                "salary_support": int(len(sal)),
                "salary_median": float(sal.median()) if len(sal) >= config.MIN_SALARY_SUPPORT else None,
                "salary_q1": float(sal.quantile(.25)) if len(sal) >= config.MIN_SALARY_SUPPORT else None,
                "salary_q3": float(sal.quantile(.75)) if len(sal) >= config.MIN_SALARY_SUPPORT else None,
                "common_roles": g["role"].dropna().value_counts().head(3).index.tolist(),
                "top_ug": g["ug_degree"].dropna().value_counts().head(2).index.tolist(),
                "top_pg": g["pg_degree"].dropna().value_counts().head(2).index.tolist(),
            }

    # ------------------------------------------------------------------ #
    # user input helpers
    def user_terms(self, skills) -> set[str]:
        return set(self._analyzer(normalise_user_skills(skills)))

    def _covered(self, term: str, terms: set[str]) -> bool:
        if term in terms:
            return True
        parts = term.split()
        return len(parts) > 1 and all(p in terms for p in parts)

    def _components(self, terms: set[str], exp: float | None, city: str | None, industry: str | None):
        """Return dict of arrays (one value per role) for the four non-ML components."""
        sk, ex, lo, it = [], [], [], []
        for role in self.classes:
            core = self.core_skills[role]
            w = core["role_share"].to_numpy()
            hit = np.array([self._covered(s, terms) for s in core["skill"]], dtype=float)
            sk.append(float((w * hit).sum() / w.sum()) if w.sum() else 0.0)
            st = self.role_stats[role]
            if exp is None or np.isnan(exp):
                ex.append(0.5)
            elif st["exp_q1"] <= exp <= st["exp_q3"]:
                ex.append(1.0)
            else:
                dist = st["exp_q1"] - exp if exp < st["exp_q1"] else exp - st["exp_q3"]
                ex.append(float(np.exp(-dist / 3.0)))
            if city and city in self.overall_city_share:
                lift = st["city_share"].get(city, 0.0) / self.overall_city_share[city]
                lo.append(float(min(1.0, lift)))
            else:
                lo.append(0.5)
            if industry and industry in self.overall_ind_share:
                lift = st["industry_share"].get(industry, 0.0) / self.overall_ind_share[industry]
                it.append(float(min(1.0, lift)))
            else:
                it.append(0.5)
        return {"skills": np.array(sk), "experience": np.array(ex),
                "location": np.array(lo), "interest": np.array(it)}

    def blend(self, ml: np.ndarray, comps: dict) -> np.ndarray:
        w = self.weights
        return (w["ml"] * ml + w["skills"] * comps["skills"] + w["experience"] * comps["experience"]
                + w["location"] * comps["location"] + w["interest"] * comps["interest"])

    # ------------------------------------------------------------------ #
    def make_features(self, profile: dict) -> pd.DataFrame:
        return pd.DataFrame([{
            "skills_text": normalise_user_skills(profile.get("skills")),
            "exp_mid": profile.get("experience_years"),
            "ug_degree": profile.get("ug_degree") or "Unknown",
            "pg_degree": profile.get("pg_degree") or "Unknown",
            "cities": profile.get("city") or "",
            "industry": profile.get("industry") or "Unknown",
        }])[FEATURE_COLUMNS]

    def gaps_for(self, role: str, terms: set[str], max_gaps: int = 10) -> list[dict]:
        core = self.core_skills[role]
        missing = [s for s in core["skill"] if not self._covered(s, terms)]
        keep = set(dedupe_terms(missing))
        rows = []
        for _, r in core.iterrows():
            if r["skill"] in keep:
                share = 100 * r["role_share"]
                rows.append({"skill": r["skill"], "role_share_pct": round(share, 1),
                             "lift": round(float(r["lift"]), 1),
                             "priority": "High" if share >= 15 else "Medium" if share >= 8 else "Low"})
        return rows[:max_gaps]

    def roadmap(self, role: str, gaps: list[dict], terms: set[str], exp: float | None) -> list[dict]:
        st = self.role_stats[role]
        high = [g["skill"] for g in gaps if g["priority"] == "High"]
        med = [g["skill"] for g in gaps if g["priority"] == "Medium"]
        low = [g["skill"] for g in gaps if g["priority"] == "Low"]
        premium_path = config.TABLE_DIR / "stats_skill_salary_premium.csv"
        prem = []
        if premium_path.exists():
            p = pd.read_csv(premium_path)
            p = p[p["significant_fdr_5pct"] & (p["exp_adjusted_premium_pct"] > 0)]
            have = {g["skill"] for g in gaps}
            prem = [s for s in p["skill"] if s in have and s not in high]
        phases, used = [], set()

        def take(cands, n):
            out = [c for c in cands if c not in used][:n]
            used.update(out)
            return out

        p1 = take(high or [g["skill"] for g in gaps], 3)
        if p1:
            phases.append({"phase": "1. Close the biggest gaps", "focus": p1,
                           "why": "Most frequently requested skills for this role that are missing from your profile."})
        p2 = take([s for s in (high + med) if s not in prem], 4)
        if p2:
            phases.append({"phase": "2. Build the core toolkit", "focus": p2,
                           "why": "Common requirements that employers list next after the essentials."})
        p3 = take(prem or low or [g["skill"] for g in gaps], 3)
        if p3:
            phases.append({"phase": "3. Stand out", "focus": p3,
                           "why": ("Skills linked to higher advertised pay after adjusting for experience "
                                   "(descriptive, not causal)." if prem else
                                   "Less common but relevant skills that add breadth.")})
        if exp is not None and not np.isnan(exp) and exp < st["exp_q1"]:
            phases.append({"phase": "Experience milestone",
                           "focus": [f"{st['exp_q1']:.1f}-{st['exp_q3']:.1f} years typical"],
                           "why": f"Typical postings ask for {st['exp_q1']:.1f}-{st['exp_q3']:.1f} years; "
                                  f"you have {exp:g}. Internships, projects or adjacent entry roles help bridge this."})
        return phases

    def reasons_for(self, role, rec_components, matched, profile, ml_p) -> list[str]:
        st = self.role_stats[role]
        out = []
        if matched:
            out.append(f"Your skills match {len(matched)} in-demand skill(s) for this role: "
                       + ", ".join(matched[:6]) + ".")
        out.append(f"The trained model gives this role a {100*ml_p:.0f}% probability for your profile "
                   f"(out of {len(self.classes)} role families).")
        exp = profile.get("experience_years")
        if exp is not None:
            if st["exp_q1"] <= exp <= st["exp_q3"]:
                out.append(f"Your {exp:g} years of experience sits inside the typical range "
                           f"({st['exp_q1']:.1f}-{st['exp_q3']:.1f} years).")
            else:
                out.append(f"Typical postings ask for {st['exp_q1']:.1f}-{st['exp_q3']:.1f} years; you have {exp:g}.")
        city = profile.get("city")
        if city and city in st["city_share"]:
            out.append(f"{100*st['city_share'][city]:.0f}% of postings for this role list {city}.")
        ind = profile.get("industry")
        if ind and ind in st["industry_share"]:
            out.append(f"{100*st['industry_share'][ind]:.0f}% of these postings are in your chosen industry ({ind}).")
        return out

    # ------------------------------------------------------------------ #
    def predict_salary(self, profile: dict, role: str) -> dict | None:
        if self.salary_model is None or not self.salary_meta.get("reliable"):
            return None
        row = self.make_features(profile).assign(role_category=role)
        pred = float(np.exp(self.salary_model.predict(row)[0]))
        sup = self.salary_meta.get("role_salary_support", {}).get(role, 0)
        return {"estimate_lpa": pred,
                "low_lpa": pred * float(np.exp(self.salary_meta["interval_log_low"])),
                "high_lpa": pred * float(np.exp(self.salary_meta["interval_log_high"])),
                "role_salary_support": int(sup),
                "low_support": sup < config.MIN_SALARY_SUPPORT}

    def recommend(self, profile: dict, top_n: int = 5) -> dict:
        terms = self.user_terms(profile.get("skills"))
        warnings = []
        if len(terms) < 3:
            warnings.append("Very few recognisable skills were entered - add more skills for a reliable result.")
        unknown = sorted(t for t in terms if t not in self._vocab and " " not in t)
        city = profile.get("city")
        if city and city not in self.overall_city_share:
            warnings.append(f"'{city}' has too few postings in the data; location was ignored.")
        ml = self.model.predict_proba(self.make_features(profile))[0]
        comps = self._components(terms, profile.get("experience_years"), city, profile.get("industry"))
        score = self.blend(ml, comps)
        recs = []
        for i in np.argsort(-score)[:top_n]:
            role = str(self.classes[i])
            st = self.role_stats[role]
            core = self.core_skills[role]
            matched = dedupe_terms([s for s in core["skill"] if self._covered(s, terms)])
            gaps = self.gaps_for(role, terms)
            recs.append({
                "role": role, "match_score": round(100 * float(score[i]), 1),
                "components": {"ml_probability": float(ml[i]), "skills": float(comps["skills"][i]),
                               "experience": float(comps["experience"][i]),
                               "location": float(comps["location"][i]),
                               "interest": float(comps["interest"][i])},
                "matched_skills": matched, "skill_gaps": gaps,
                "reasons": self.reasons_for(role, comps, matched, profile, float(ml[i])),
                "roadmap": self.roadmap(role, gaps, terms, profile.get("experience_years")),
                "stats": st, "salary": self.predict_salary(profile, role),
            })
        return {"profile": profile, "recommendations": recs, "warnings": warnings,
                "unrecognised_terms": unknown[:10], "n_recognised_terms": len(terms)}

    # ------------------------------------------------------------------ #
    # data-driven option lists for the UI
    def options(self) -> dict:
        d = self.d
        cities = [c for c, s in sorted(self.overall_city_share.items(), key=lambda x: -x[1])
                  if s * self.n_docs >= 30]
        return {
            "cities": cities,
            "industries": d["industry"].value_counts().head(25).index.tolist(),
            "ug": d["ug_degree"].dropna().value_counts().head(10).index.tolist(),
            "pg": d["pg_degree"].dropna().value_counts().head(10).index.tolist(),
            "skills": self.skill_index.top_overall(80)["skill"].tolist(),
        }


# --------------------------------------------------------------------------- #
# Offline evaluation on the held-out TEST companies (train-only fit)
# --------------------------------------------------------------------------- #

def evaluate(log=print) -> pd.DataFrame:
    from .train_models import fit_pipeline, make_pipeline
    d = prepare_role_dataset(load_clean())
    split = pd.read_csv(config.TABLE_DIR / "ml_split.csv").set_index("job_id")["split"]
    is_test = d["job_id"].map(split).eq("test").to_numpy()
    train, test = d[~is_test].reset_index(drop=True), d[is_test].reset_index(drop=True)
    meta = json.load(open(config.MODELS_DIR / "role_model_metadata.json"))
    pipe = fit_pipeline(make_pipeline(meta["model_name"], meta["params"]),
                        train[FEATURE_COLUMNS], train[TARGET], meta["params"]["balanced"])
    rec = CareerRecommender(train, pipe)
    ml = pipe.predict_proba(test[FEATURE_COLUMNS])
    comp_rows = [rec._components(rec.user_terms(r.skills_text), r.exp_mid,
                                 (r.cities.split("|")[0] if r.cities else None), r.industry)
                 for r in test.itertuples()]
    comps = {k: np.vstack([c[k] for c in comp_rows]) for k in comp_rows[0]}
    truth = np.array([list(rec.classes).index(t) for t in test[TARGET]])

    def rank_metrics(S):
        order = np.argsort(-S, axis=1)
        ranks = np.array([np.where(order[i] == truth[i])[0][0] + 1 for i in range(len(truth))])
        return {"top1": np.mean(ranks <= 1), "top3": np.mean(ranks <= 3), "top5": np.mean(ranks <= 5),
                "mean_reciprocal_rank": np.mean(1 / ranks)}

    variants = {
        "ML probability only": ml,
        "Skill coverage only": comps["skills"],
        "Blend (default weights)": rec.blend(ml, comps),
        "Blend without location/interest": rec.blend(ml, {**comps, "location": comps["location"] * 0 + .5,
                                                          "interest": comps["interest"] * 0 + .5}),
    }
    out = pd.DataFrame([{"ranking": k, **rank_metrics(v)} for k, v in variants.items()])
    out["n_test_profiles"] = len(test)
    out.to_csv(config.TABLE_DIR / "recommender_eval.csv", index=False)
    log(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    evaluate()
