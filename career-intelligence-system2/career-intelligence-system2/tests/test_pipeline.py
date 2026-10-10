"""Run with:  pytest -q   (needs the pipeline outputs: python run_pipeline.py)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.features import FEATURE_COLUMNS, TARGET, grouped_stratified_split, prepare_role_dataset
from src.preprocessing import (clean_skill_text, load_clean, normalise_degree, parse_education,
                               parse_experience, parse_locations, parse_salary_lpa)
from src.statistics_analysis import _holm, descriptive_stats

needs_artifacts = pytest.mark.skipif(not (config.MODELS_DIR / "role_model.joblib").exists(),
                                     reason="run python run_pipeline.py first")


# ---------------------------------------------------------------- parsing
def test_parse_experience():
    assert parse_experience("2 - 7 yrs") == (2.0, 7.0)
    assert all(np.isnan(parse_experience("Not Mentioned")))


def test_parse_salary():
    assert parse_salary_lpa("3,00,000 - 6,00,000 P.A. Best in Industry") == (3.0, 6.0)
    assert all(np.isnan(parse_salary_lpa("Not Disclosed by Recruiter")))
    assert all(np.isnan(parse_salary_lpa("10000 to 45000 per month")))


def test_parse_locations_and_education():
    assert parse_locations("Bengaluru/Bangalore , Bengaluru / Bangalore , Gurgaon , Noida") == ["Bengaluru", "Delhi NCR"]
    assert parse_education("UG: B.Tech/B.E. - Computers PG:M.Tech - Any Specialization") == ("B.Tech/B.E.", "M.Tech")
    assert normalise_degree("Any Graduate - Any Specialization, Graduation Not Required") == "Any Graduate"


def test_skill_cleaning_keeps_special_tokens():
    out = clean_skill_text("C++, C#, .NET / ASP.NET & Node.js.")
    for tok in ["c++", "c#", ".net", "asp.net", "node.js"]:
        assert tok in out.split()


# ---------------------------------------------------------------- data + leakage
def test_clean_data_sane():
    df = load_clean()
    assert df["job_id"].is_unique
    assert df["salary_mid_lpa"].dropna().between(config.MIN_SALARY_LPA, config.MAX_SALARY_LPA).all()
    assert (df["exp_min"].dropna() <= df["exp_max"].dropna()).all()


def test_no_label_columns_in_features():
    banned = {"role_category", "role", "functional_area", "job_title", "jobdescription"}
    assert not banned & set(FEATURE_COLUMNS)


def test_split_has_no_company_overlap():
    d = prepare_role_dataset(load_clean())
    tr, te = grouped_stratified_split(d)
    assert not set(d["group"].iloc[tr]) & set(d["group"].iloc[te])
    assert d[TARGET].nunique() >= 10


# ---------------------------------------------------------------- statistics
def test_descriptive_matches_pandas():
    df = pd.DataFrame({"x": [1.0, 2, 2, 3, 10]})
    r = descriptive_stats(df, ["x"]).iloc[0]
    assert r["mean"] == pytest.approx(df.x.mean()) and r["variance"] == pytest.approx(df.x.var(ddof=1))
    assert r["mode"] == 2 and r["Q1"] == df.x.quantile(.25)


def test_holm_is_valid():
    p = [0.01, 0.04, 0.03]
    adj = _holm(p)
    assert all(a >= b for a, b in zip(adj, p)) and all(0 <= a <= 1 for a in adj)


# ---------------------------------------------------------------- recommender + guidance
@pytest.fixture(scope="module")
def rec():
    from src.recommender import CareerRecommender
    return CareerRecommender.from_artifacts()


@needs_artifacts
def test_recommender_output(rec):
    from src.recommender import WEIGHTS
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    res = rec.recommend({"skills": "java, j2ee, spring, sql", "experience_years": 4, "ug_degree": "B.Tech/B.E.",
                         "pg_degree": "Post Graduation Not Required", "city": "Bengaluru",
                         "industry": "IT-Software / Software Services"}, top_n=5)
    recs = res["recommendations"]
    scores = [r["match_score"] for r in recs]
    assert len(recs) == 5 and scores == sorted(scores, reverse=True) and all(0 <= s <= 100 for s in scores)
    assert "Programming & Design" in [r["role"] for r in recs[:3]]
    have = rec.user_terms("java, j2ee, spring, sql")
    assert not {g["skill"] for g in recs[0]["skill_gaps"]} & have
    flat = [s for p in recs[0]["roadmap"] for s in p["focus"] if "years" not in s]
    assert len(flat) == len(set(flat))          # no skill repeated across roadmap phases


@needs_artifacts
def test_guidance_uses_results(rec):
    from src.guidance import answer_question, build_guidance
    res = rec.recommend({"skills": "tally, gst, accounting", "experience_years": 3, "ug_degree": "B.Com"})
    g = build_guidance(res)
    assert res["recommendations"][0]["role"] in g["headline"]
    assert "probability" in answer_question("why this role", res).lower()
    assert "lakh" in answer_question("what is the salary", res).lower() or "salary" in answer_question("salary", res).lower()


@needs_artifacts
def test_artifact_metadata():
    m = json.load(open(config.MODELS_DIR / "role_model_metadata.json"))
    s = json.load(open(config.MODELS_DIR / "salary_model_metadata.json"))
    assert len(m["classes"]) >= 10 and 0 < m["test_metrics"]["accuracy"] < 1
    assert "reliable" in s and s["interval_log_low"] < 0 < s["interval_log_high"]
