"""Module 5 - Applied AI: personalised career guidance grounded in model results.

This is NOT a free-text chatbot.  Every sentence is assembled from numbers the
recommender, the role model, the salary model or the evaluation tables actually
produced for this user.  Nothing is generated from outside knowledge, so the
guidance cannot invent facts; if a number is unavailable the sentence is skipped.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config


def _eval_top3() -> float | None:
    p = config.TABLE_DIR / "recommender_eval.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p)
    row = e[e["ranking"].str.startswith("Blend (default")]
    return float(row["top3"].iloc[0]) if len(row) else None


def build_guidance(result: dict) -> dict:
    recs = result["recommendations"]
    if not recs:
        return {"headline": "No recommendation available.", "sections": [], "next_actions": [], "caveats": []}
    top = recs[0]
    st = top["stats"]
    sections, actions = [], []

    # ---- confidence ------------------------------------------------------- #
    gap = top["match_score"] - recs[1]["match_score"] if len(recs) > 1 else 99
    if gap >= 8:
        conf = f"{top['role']} is a clear first choice ({gap:.0f} points ahead of the next role)."
    elif gap >= 3:
        conf = (f"{top['role']} leads, but {recs[1]['role']} is fairly close ({gap:.0f} points behind) - "
                "worth exploring both.")
    else:
        conf = (f"Several roles score almost equally ({top['role']} and {recs[1]['role']} differ by "
                f"{gap:.1f} points), so your profile fits more than one path.")
    sections.append(("Overall picture", conf))

    # ---- strengths and weak spots ------------------------------------------ #
    c = top["components"]
    strengths = []
    if top["matched_skills"]:
        strengths.append("skills already in demand for this role: " + ", ".join(top["matched_skills"][:6]))
    if c["experience"] >= 0.99:
        strengths.append("your experience is inside the typical range")
    if c["location"] >= 0.99 and result["profile"].get("city"):
        strengths.append(f"the role is well represented in {result['profile']['city']}")
    weak = []
    if c["skills"] < 0.35:
        weak.append(f"you cover only {100*c['skills']:.0f}% of the role's most requested skills")
    if c["experience"] < 0.7:
        weak.append(f"typical postings ask for {st['exp_q1']:.1f}-{st['exp_q3']:.1f} years of experience")
    txt = ""
    if strengths:
        txt += "Strengths: " + "; ".join(strengths) + ". "
    if weak:
        txt += "Main gaps: " + "; ".join(weak) + "."
    if txt:
        sections.append((f"Your fit for {top['role']}", txt.strip()))

    # ---- pay ---------------------------------------------------------------- #
    pay = []
    if top.get("salary"):
        s = top["salary"]
        pay.append(f"The salary model estimates about {s['estimate_lpa']:.1f} lakh per year for your profile "
                   f"(80% range {s['low_lpa']:.1f}-{s['high_lpa']:.1f}).")
        if s["low_support"]:
            pay.append(f"Caution: only {s['role_salary_support']} salaried postings exist for this role, so treat it as rough.")
    if st["salary_median"] is not None:
        pay.append(f"Across {st['salary_support']} postings that disclose pay, the median for this role is "
                   f"{st['salary_median']:.1f} lakh (middle half {st['salary_q1']:.1f}-{st['salary_q3']:.1f}).")
    if pay:
        sections.append(("Pay expectations", " ".join(pay)))

    # ---- alternatives -------------------------------------------------------- #
    if len(recs) > 1:
        easiest = max(recs[1:], key=lambda r: r["components"]["skills"])
        alt = None if easiest["components"]["skills"] < 0.10 else f"Among the other suggestions, closest to your current skills: {easiest['role']} ({100*easiest['components']['skills']:.0f}% skill coverage)."
        if alt:
            sections.append(("Alternatives", alt))

    # ---- next actions --------------------------------------------------------- #
    first = [g["skill"] for g in top["skill_gaps"] if g["priority"] == "High"][:3] or \
            [g["skill"] for g in top["skill_gaps"][:3]]
    if first:
        actions.append("Start with these skills: " + ", ".join(first) + ".")
    if c["experience"] < 0.7:
        actions.append("Look for internships, projects or entry-level postings to build experience toward "
                       f"{st['exp_q1']:.1f}+ years.")
    if result.get("n_recognised_terms", 0) < 5:
        actions.append("Add more specific skills to your profile - the recommendation improves with detail.")
    if len(recs) > 1 and gap < 3:
        actions.append(f"Compare the skill gaps of {top['role']} and {recs[1]['role']} before committing.")
    if result.get("unrecognised_terms"):
        actions.append("Some terms were not found in the job data (" + ", ".join(result["unrecognised_terms"][:5])
                       + "); they did not influence the result.")

    caveats = ["Data comes from Naukri.com postings of 2015-2016; demand and pay have changed since."]
    t3 = _eval_top3()
    if t3 is not None:
        caveats.append(f"On held-out postings the true role family was inside the top 3 suggestions {100*t3:.0f}% of the time.")
    caveats.append("The match score ranks fit; it does not predict hiring outcomes.")
    return {"headline": f"Best match: {top['role']} ({top['match_score']:.0f}/100)",
            "sections": sections, "next_actions": actions, "caveats": caveats}


def render_markdown(g: dict) -> str:
    lines = [f"### {g['headline']}"]
    for title, body in g["sections"]:
        lines.append(f"**{title}.** {body}")
    if g["next_actions"]:
        lines.append("**Suggested next steps**\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(g["next_actions"], 1)))
    lines.append("_" + " ".join(g["caveats"]) + "_")
    return "\n\n".join(lines)


def answer_question(question: str, result: dict) -> str:
    """Keyword-routed Q&A that answers only from the recommendation result."""
    q = question.lower()
    recs = result["recommendations"]
    if not recs:
        return "Please run a recommendation first."
    top = recs[0]
    st = top["stats"]
    if any(k in q for k in ("why", "reason", "explain")):
        return " ".join(top["reasons"])
    if any(k in q for k in ("learn", "gap", "skill", "roadmap", "study", "improve")):
        return "; ".join(f"{p['phase']}: {', '.join(p['focus'])}" for p in top["roadmap"]) or "No skill gaps found."
    if any(k in q for k in ("salary", "pay", "earn", "ctc", "lakh")):
        parts = []
        if top.get("salary"):
            s = top["salary"]
            parts.append(f"Model estimate {s['estimate_lpa']:.1f} lakh/yr (80% range {s['low_lpa']:.1f}-{s['high_lpa']:.1f}).")
        if st["salary_median"] is not None:
            parts.append(f"Median advertised pay for the role: {st['salary_median']:.1f} lakh ({st['salary_support']} postings).")
        return " ".join(parts) or "Not enough salary data for this role."
    if any(k in q for k in ("city", "cities", "location", "where", "place")):
        top_c = sorted(st["city_share"].items(), key=lambda x: -x[1])[:3]
        return "Most postings for this role list: " + ", ".join(f"{c} ({100*s:.0f}%)" for c, s in top_c) + "."
    if any(k in q for k in ("other", "alternative", "compare", "second", "instead")):
        return "; ".join(f"{r['role']} ({r['match_score']:.0f}/100)" for r in recs[1:4])
    if any(k in q for k in ("experience", "fresher", "years")):
        return f"Typical experience for {top['role']}: {st['exp_q1']:.1f}-{st['exp_q3']:.1f} years (median {st['exp_median']:.1f})."
    return ("I can answer from your results: ask why this role, what to learn, salary, cities, "
            "alternatives or experience.")
