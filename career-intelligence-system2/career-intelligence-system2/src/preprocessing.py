"""Module 1a - Load the raw Naukri.com dataset and turn it into a clean table.

What this file does (in plain words)
------------------------------------
1. Reads the raw CSV (downloads it first if it is missing).
2. Removes completely blank rows and duplicate postings.
3. Pulls structured fields out of the free-text ``jobdescription`` column
   (Role Category, Role, Functional Area, Key skills).
4. Converts text such as "2 - 7 yrs" or "3,00,000 - 6,00,000 P.A" into numbers.
5. Saves a tidy file and a data-quality report with the *real* counts.

No statistics or models are created here, so nothing can leak into them.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# 0. Getting the file
# --------------------------------------------------------------------------- #

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw_data(verbose: bool = True) -> Path:
    """Return the raw file path, downloading it from GitHub if needed."""
    path = config.RAW_PATH
    if not path.exists():
        if verbose:
            print(f"Downloading dataset from {config.DATA_URL} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.DATA_URL, path)
    digest = sha256_of(path)
    if digest != config.DATA_SHA256 and verbose:
        print("WARNING: file checksum differs from the one used to build this "
              "project. Results may differ slightly.")
    return path


# --------------------------------------------------------------------------- #
# 1. Small parsing helpers (each one is easy to unit-test)
# --------------------------------------------------------------------------- #

def _first_match(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.S)
    return m.group(1).strip() if m else None


def parse_description(text: str) -> dict:
    """Extract the structured tail of a Naukri job description."""
    text = text if isinstance(text, str) else ""
    skills = _first_match(
        r"Keyskills\s*(.*?)\s*(?:Desired Candidate Profile|Company Profile|"
        r"Download PPT|View Contact Details|$)", text)
    return {
        "functional_area": _first_match(r"Functional Area:\s*(.*?)\s+Role Category:", text),
        "role_category": _first_match(r"Role Category:\s*(.*?)\s+Role:", text),
        "role": _first_match(r"\sRole:\s*(.*?)\s+Keyskills", text),
        "key_skills_raw": skills,
    }


def parse_experience(text) -> tuple[float, float]:
    """'2 - 7 yrs' -> (2.0, 7.0). Anything else (e.g. 'Not Mentioned') -> NaN."""
    if not isinstance(text, str):
        return (np.nan, np.nan)
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*yr", text, flags=re.I)
    if not m:
        return (np.nan, np.nan)
    lo, hi = float(m.group(1)), float(m.group(2))
    if hi < lo or hi > 40:
        return (np.nan, np.nan)
    return (lo, hi)


def parse_salary_lpa(text) -> tuple[float, float]:
    """'3,00,000 - 6,00,000 P.A ...' -> (3.0, 6.0) in INR lakh per annum.

    Only a clean "N - N P.A" range at the start of the field is trusted.
    Free-text pay ("Best in industry", "upto 30k", "per month" ...) -> NaN.
    """
    if not isinstance(text, str):
        return (np.nan, np.nan)
    m = re.match(r"\s*(\d[\d,]*)\s*-\s*(\d[\d,]*)\s*P\.?A", text, flags=re.I)
    if not m:
        return (np.nan, np.nan)
    lo = float(m.group(1).replace(",", "")) / 1e5
    hi = float(m.group(2).replace(",", "")) / 1e5
    return (lo, hi)


_CITY_RULES = [
    (r"bengaluru|bangalore", "Bengaluru"),
    (r"hyderabad|secunderabad", "Hyderabad"),
    (r"delhi|ncr|gurgaon|gurugram|noida|ghaziabad|faridabad", "Delhi NCR"),
    (r"mumbai|thane|navi mumbai", "Mumbai"),
    (r"pune", "Pune"),
    (r"chennai", "Chennai"),
    (r"kolkata|calcutta", "Kolkata"),
    (r"ahmedabad", "Ahmedabad"),
    (r"jaipur", "Jaipur"),
    (r"chandigarh|mohali", "Chandigarh"),
    (r"coimbatore", "Coimbatore"),
    (r"kochi|cochin|ernakulam", "Kochi"),
    (r"trivandrum|thiruvananthapuram", "Trivandrum"),
]


def normalise_city(raw: str) -> str | None:
    s = raw.strip().lower()
    if not s:
        return None
    for pattern, name in _CITY_RULES:
        if re.search(pattern, s):
            return name
    return raw.strip().title()


def parse_locations(text) -> list[str]:
    """Split 'Bengaluru/Bangalore , Chennai' into unique normalised cities."""
    if not isinstance(text, str):
        return []
    cities: list[str] = []
    for part in text.split(","):
        city = normalise_city(part)
        if city and city not in cities:
            cities.append(city)
    return cities


def normalise_degree(segment) -> str | None:
    """'B.Tech/B.E. - Computers, Graduation Not Required' -> 'B.Tech/B.E.'"""
    if not isinstance(segment, str) or not segment.strip():
        return None
    first = segment.split(",")[0]
    first = re.split(r"\s+-\s+", first)[0].strip()
    return first or None


def parse_education(text) -> tuple[str | None, str | None]:
    """Return (first accepted UG degree, first accepted PG degree)."""
    if not isinstance(text, str):
        return (None, None)
    ug = _first_match(r"UG:\s*(.*?)(?:\s+PG:|\s+Doctorate:|$)", text)
    pg = _first_match(r"PG:\s*(.*?)(?:\s+Doctorate:|$)", text)
    return (normalise_degree(ug), normalise_degree(pg))


def clean_skill_text(raw) -> str:
    """Lower-case and tidy skill text.

    Keeps letters, digits and the symbols used in skill names (+ # .) so that
    'c++', 'c#', '.net' and 'node.js' survive; drops stray punctuation.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.lower()
    s = re.sub(r"[^a-z0-9+#.\s]", " ", s)          # '/', '&', ',', '-' ... -> space
    tokens = []
    for tok in s.split():
        tok = tok.rstrip(".")                       # sentence-ending dots
        if re.search(r"[a-z0-9]", tok):
            tokens.append(tok)
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# 2. Main cleaning routine
# --------------------------------------------------------------------------- #

def clean_jobs(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean dataframe, data-quality report)."""
    report: dict = {"raw_rows": int(len(raw))}

    df = raw.dropna(how="all").copy()
    report["blank_rows_removed"] = report["raw_rows"] - len(df)

    n0 = len(df)
    df = df.drop_duplicates(subset="uniq_id")
    df = df.drop_duplicates(subset=["company", "jobtitle", "jobdescription"])
    report["duplicate_rows_removed"] = int(n0 - len(df))
    report["rows_after_dedup"] = int(len(df))

    parsed = pd.DataFrame([parse_description(t) for t in df["jobdescription"]],
                          index=df.index)
    exp = pd.DataFrame([parse_experience(t) for t in df["experience"]],
                       index=df.index, columns=["exp_min", "exp_max"])
    sal = pd.DataFrame([parse_salary_lpa(t) for t in df["payrate"]],
                       index=df.index, columns=["salary_min_lpa", "salary_max_lpa"])
    edu = pd.DataFrame([parse_education(t) for t in df["education"]],
                       index=df.index, columns=["ug_degree", "pg_degree"])
    cities = [parse_locations(t) for t in df["joblocation_address"]]

    out = pd.DataFrame(index=df.index)
    out["job_id"] = df["uniq_id"]
    out["company"] = df["company"].fillna("Unknown").str.strip()
    out["job_title"] = df["jobtitle"].fillna("").str.strip()
    out["industry"] = df["industry"].fillna("Unknown").str.strip()
    out["functional_area"] = parsed["functional_area"]
    out["role_category"] = parsed["role_category"].str.strip()
    out["role"] = parsed["role"]
    out["skills_text"] = parsed["key_skills_raw"].map(clean_skill_text)
    out["n_skill_words"] = out["skills_text"].str.split().str.len().fillna(0).astype(int)

    out = pd.concat([out, exp, edu], axis=1)
    out["exp_mid"] = (out["exp_min"] + out["exp_max"]) / 2

    out["cities"] = ["|".join(c) for c in cities]
    out["primary_city"] = [c[0] if c else None for c in cities]
    out["n_cities"] = [len(c) for c in cities]

    # --- salary: parse, then drop impossible values ------------------------- #
    out = pd.concat([out, sal], axis=1)
    out["salary_mid_lpa"] = (out["salary_min_lpa"] + out["salary_max_lpa"]) / 2
    parsed_n = int(out["salary_mid_lpa"].notna().sum())
    bad = (
        (out["salary_min_lpa"] > out["salary_max_lpa"])
        | (out["salary_mid_lpa"] < config.MIN_SALARY_LPA)
        | (out["salary_mid_lpa"] > config.MAX_SALARY_LPA)
    )
    for c in ["salary_min_lpa", "salary_max_lpa", "salary_mid_lpa"]:
        out.loc[bad, c] = np.nan
    out["salary_disclosed"] = out["salary_mid_lpa"].notna()
    report["salary_ranges_parsed"] = parsed_n
    report["salary_removed_as_implausible"] = int(bad.sum())

    out["post_date"] = pd.to_datetime(df["postdate"], errors="coerce", utc=True)
    out["post_year"] = out["post_date"].dt.year
    out["post_date"] = out["post_date"].dt.tz_localize(None)
    out = out.reset_index(drop=True)

    # ---------------------------------------------------------------- report
    report.update({
        "rows_clean": int(len(out)),
        "rows_with_role_category": int(out["role_category"].notna().sum()),
        "rows_with_skills": int((out["skills_text"] != "").sum()),
        "rows_with_experience": int(out["exp_mid"].notna().sum()),
        "rows_with_location": int((out["n_cities"] > 0).sum()),
        "rows_with_education": int(out["ug_degree"].notna().sum()),
        "rows_with_salary": int(out["salary_disclosed"].sum()),
        "salary_disclosure_rate_pct": round(100 * out["salary_disclosed"].mean(), 2),
        "unique_companies": int(out["company"].nunique()),
        "unique_role_categories": int(out["role_category"].nunique()),
        "post_year_counts": {str(k): int(v) for k, v in
                             out["post_year"].value_counts().sort_index().items()},
    })
    return out, report


def build_clean_dataset(save: bool = True, verbose: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(ensure_raw_data(verbose))
    clean, report = clean_jobs(raw)
    if save:
        clean.to_csv(config.PROCESSED_PATH, index=False)
        with open(config.TABLE_DIR / "data_quality_report.json", "w") as f:
            json.dump(report, f, indent=2)
    if verbose:
        print(json.dumps(report, indent=2))
    return clean


def load_clean() -> pd.DataFrame:
    """Load the cleaned file (build it first if it does not exist)."""
    if not config.PROCESSED_PATH.exists():
        return build_clean_dataset(verbose=False)
    df = pd.read_csv(config.PROCESSED_PATH, parse_dates=["post_date"])
    df["skills_text"] = df["skills_text"].fillna("")
    df["cities"] = df["cities"].fillna("")
    return df


if __name__ == "__main__":
    build_clean_dataset()
