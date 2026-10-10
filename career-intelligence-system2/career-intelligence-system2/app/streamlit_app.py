"""Streamlit dashboard - AI-Powered Career & Job Market Intelligence System (SDG 8).

Run:  streamlit run app/streamlit_app.py
All numbers shown are read from files produced by the pipeline (reports/, models/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.guidance import answer_question, build_guidance, render_markdown
from src.preprocessing import load_clean
from src.skills import SkillIndex

st.set_page_config(page_title="Career & Job Market Intelligence", page_icon="🧭", layout="wide")
st.markdown("""<style>
.block-container{padding-top:0.9rem;max-width:1250px}
[data-testid="stVerticalBlock"]{gap:0.7rem}
[data-testid="stSidebar"] .block-container{padding-top:1rem}
div[data-testid="stMetric"]{background:#f4f8fa;border:1px solid #e1ebf0;border-radius:10px;padding:10px 14px}
.role-card{border:1px solid #d9e4ea;border-left:6px solid #1f6f8b;border-radius:8px;padding:10px 16px;margin-bottom:6px;background:#fff}
</style>""", unsafe_allow_html=True)
BLUE, ORANGE = "#1f6f8b", "#e07a1f"


# ------------------------------------------------------------------ loaders
def need_artifacts() -> bool:
    return not (config.MODELS_DIR / "role_model.joblib").exists() or not (config.TABLE_DIR / "ml_test_metrics.csv").exists()


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.TABLE_DIR / f"{name}.csv")


@st.cache_data
def jobs() -> pd.DataFrame:
    return load_clean()


@st.cache_data
def kpis() -> dict:
    return json.load(open(config.TABLE_DIR / "overview_kpis.json"))


@st.cache_data
def quality() -> dict:
    return json.load(open(config.TABLE_DIR / "data_quality_report.json"))


@st.cache_data
def meta(name: str) -> dict:
    return json.load(open(config.MODELS_DIR / name))


@st.cache_resource
def recommender():
    from src.recommender import CareerRecommender
    return CareerRecommender.from_artifacts()


@st.cache_resource
def skill_index():
    d = jobs()
    d = d[(d["skills_text"] != "") & d["role_category"].notna()]
    return SkillIndex().fit(d["skills_text"], d["role_category"])


def bar(df, x, y, title, color=BLUE, orient="h", **kw):
    fig = px.bar(df, x=x, y=y, orientation=orient, title=title, color_discrete_sequence=[color], **kw)
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(autorange="reversed") if orient == "h" else {})
    return fig


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("🧭 AI-Powered Career & Job Market Intelligence System")
    st.caption("SDG 8 - Decent Work and Economic Growth | Real data: Naukri.com job postings (2015-2016)")
    k = kpis()
    c = st.columns(4)
    c[0].metric("Job postings analysed", f"{k['total_postings']:,}")
    c[1].metric("Employers", f"{k['unique_companies']:,}")
    c[2].metric("Role categories", k["role_categories"])
    c[3].metric("Industries", k["industries"])
    st.subheader("How this project connects to SDG 8")
    st.write("SDG 8 asks for productive employment and decent work for all (target 8.5) and fewer young people "
             "without work or training (8.6). Job-market data can show *where* opportunities are, *what skills* "
             "unlock them and *how transparent* employers are about pay. These indicators are computed from the data:")
    h6 = table("stats_hypothesis_tests")
    h6 = h6[h6["id"] == "H6"]["interpretation"].iloc[0]
    ind = pd.DataFrame([
        {"SDG-8 lens": "Entry-level access (8.6)", "Indicator": "Postings open to freshers (0 yrs minimum)", "Value": f"{k['fresher_open_pct']:.1f}%"},
        {"SDG-8 lens": "Pay transparency (8.5)", "Indicator": "Postings that disclose a salary range", "Value": f"{k['salary_disclosed_pct']:.1f}%"},
        {"SDG-8 lens": "Regional balance", "Indicator": "Share of postings in the top 3 cities", "Value": f"{k['top3_city_share_pct']:.1f}%"},
        {"SDG-8 lens": "Sector diversification", "Indicator": f"Share in the largest industry ({k['top_industry']})", "Value": f"{k['top_industry_share_pct']:.1f}%"},
        {"SDG-8 lens": "Typical requirement", "Indicator": "Median minimum experience asked (years)", "Value": f"{k['median_exp_min']:.0f}"},
    ])
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(ind, hide_index=True, width="stretch")
        st.info("Statistical test on transparency (H6): " + h6)
    with right:
        st.subheader("What the system does")
        st.markdown("1. **Analyse** the job market (EDA + statistics)  \n2. **Predict** suitable role families with ML  \n"
                    "3. **Recommend** roles with match scores, skill gaps and a roadmap  \n"
                    "4. **Guide** the user using only the computed results  \n5. **Estimate pay** when the data supports it")
    with st.expander("Data quality report (computed)"):
        st.json(quality())
    st.warning("Limitations: postings are from 2015-2016; only about 3 in 10 disclose pay; most postings are in a few "
               "large cities; results describe the dataset, not today's whole job market.")


def page_analytics():
    st.header("📊 Job Market Analytics")
    t = st.tabs(["Roles", "Industries & cities", "Experience", "Skills", "Salary"])
    with t[0]:
        n = st.slider("Number of roles", 5, 25, 15)
        r = table("eda_role_summary").head(n)
        st.plotly_chart(bar(r, "postings", "role_category", "Postings by role category"), width="stretch")
        st.dataframe(r.round(2), hide_index=True, width="stretch")
    with t[1]:
        a, b = st.columns(2)
        a.plotly_chart(bar(table("eda_industry_summary").head(12), "postings", "industry", "Top industries", ORANGE), width="stretch")
        b.plotly_chart(bar(table("eda_city_summary").head(10), "postings", "primary_city", "Top cities", "#2e8b57"), width="stretch")
        m = table("eda_role_city_matrix_pct").set_index("role_category")
        fig = px.imshow(m, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                        title="Where each role is posted (% of the role's postings, by city)")
        st.plotly_chart(fig, width="stretch")
    with t[2]:
        e = table("eda_experience_summary")
        a, b = st.columns(2)
        a.plotly_chart(bar(e, "exp_band", "share_pct", "Share of postings by experience band", BLUE, "v"), width="stretch")
        b.plotly_chart(bar(e.dropna(subset=["median_salary_lpa"]), "exp_band", "median_salary_lpa",
                           "Median salary (lakh) by experience band", ORANGE, "v"), width="stretch")
    with t[3]:
        idx = skill_index()
        roles = ["All roles"] + table("eda_role_summary")["role_category"].tolist()
        sel = st.selectbox("Role", roles)
        if sel == "All roles":
            s = idx.top_overall(25).rename(columns={"share_pct": "share"})
            st.plotly_chart(bar(s, "share", "skill", "Most requested skill terms (% of postings with skills)"), width="stretch")
        else:
            p = idx.role_profile(sel, top=20)
            p["share_pct"] = 100 * p["role_share"]
            st.plotly_chart(bar(p, "share_pct", "skill", f"Skills requested for {sel} (% of its postings)"), width="stretch")
            st.caption("'core' = at least 1.3x more common in this role than overall (see `kind` in table).")
            st.dataframe(p[["skill", "share_pct", "lift", "kind"]].round(2), hide_index=True, width="stretch")
    with t[4]:
        d = jobs().dropna(subset=["salary_mid_lpa"])
        st.caption(f"Based on {len(d):,} postings that disclose a salary range (midpoint, INR lakh per year).")
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(d, x="salary_mid_lpa", nbins=50, color_discrete_sequence=[BLUE], title="Salary distribution"), width="stretch")
        rs = table("eda_role_summary").dropna(subset=["median_salary_lpa"]).sort_values("median_salary_lpa", ascending=False).head(12)
        b.plotly_chart(bar(rs, "median_salary_lpa", "role_category", f"Median salary by role (n >= {config.MIN_SALARY_SUPPORT})", ORANGE), width="stretch")


def page_stats():
    st.header("📐 Statistical Analysis")
    t = st.tabs(["Descriptive", "Outliers & normality", "Correlation", "Hypothesis tests", "Skills & pay"])
    with t[0]:
        st.dataframe(table("stats_descriptive").round(3), hide_index=True, width="stretch")
        st.caption("Variance and standard deviation use n-1. Skewness > 0 means a long right tail.")
        st.dataframe(table("stats_salary_by_experience_band").round(2), hide_index=True, width="stretch")
    with t[1]:
        st.dataframe(table("stats_outliers").round(2), hide_index=True, width="stretch")
        st.caption("IQR rule: outside Q1-1.5*IQR .. Q3+1.5*IQR. High salary outliers are real senior roles, so they are kept; "
                   "models use log-salary to limit their influence.")
        n = table("stats_normality")
        st.dataframe(n.round(4), hide_index=True, width="stretch")
        st.caption("Raw salary is clearly non-normal, so non-parametric tests are used for salary comparisons.")
    with t[2]:
        kind = st.radio("Method", ["spearman", "pearson"], horizontal=True)
        c = table(f"stats_correlation_{kind}").set_index("index")
        st.plotly_chart(px.imshow(c, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto"), width="stretch")
        st.dataframe(table("stats_correlation_tests").round(4), hide_index=True, width="stretch")
    with t[3]:
        h = table("stats_hypothesis_tests")
        for _, r in h.iterrows():
            with st.expander(f"{r['id']}: {r['question']}", expanded=r["id"] == "H1"):
                st.write(f"**H0:** {r['H0']}  \n**Test:** {r['test']}")
                a, b, c = st.columns(3)
                a.metric("p-value", f"{r['p_value']:.2e}")
                b.metric("Holm-adjusted p", f"{r['p_holm']:.2e}")
                c.metric(r["effect_name"], f"{r['effect_size']:.3f}", r["effect_label"], delta_color="off")
                st.write(r["interpretation"])
    with t[4]:
        p = table("stats_skill_salary_premium")
        only = st.checkbox("Only statistically significant (FDR 5%)", value=True)
        q = p[p["significant_fdr_5pct"]] if only else p
        top = pd.concat([q.head(10), q.tail(10)]).drop_duplicates("skill")
        st.plotly_chart(bar(top, "exp_adjusted_premium_pct", "skill", "Experience-adjusted salary premium (%) of postings asking for the skill"), width="stretch")
        st.dataframe(q.round(3), hide_index=True, width="stretch")
        st.caption("Descriptive association after removing the experience effect - not proof that learning a skill raises pay.")


def page_ml():
    st.header("🤖 ML Performance - Job Role Prediction")
    m = meta("role_model_metadata.json")
    st.write(f"**Task:** predict one of **{len(m['classes'])} role families** from skills, experience, education, "
             f"location and industry interest. **Selected model:** {m['model_name']} (chosen by cross-validated macro-F1).")
    tm = table("ml_test_metrics")
    cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "top3_accuracy", "cv_f1_macro"]
    st.subheader("Held-out test results (unseen companies)")
    st.dataframe(tm[cols].round(3), hide_index=True, width="stretch")
    pm = tm[~tm["model"].str.startswith("Baseline")].melt(id_vars="model", value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    st.plotly_chart(px.bar(pm, x="model", y="value", color="variable", barmode="group", range_y=[0, 1]), width="stretch")
    ci = tm[~tm["model"].str.startswith("Baseline")][["model", "accuracy_ci_low", "accuracy_ci_high", "f1_macro_ci_low", "f1_macro_ci_high"]]
    with st.expander("95% bootstrap confidence intervals (resampling companies)"):
        st.dataframe(ci.round(3), hide_index=True, width="stretch")
    st.subheader("Confusion matrix")
    names = tm[~tm["model"].str.startswith("Baseline")]["model"].tolist()
    sel = st.selectbox("Model", names, index=names.index(m["model_name"]))
    cm = pd.read_csv(config.TABLE_DIR / f"ml_confusion_{sel.replace(' ', '_').lower()}.csv", index_col=0)
    norm = cm.div(cm.sum(axis=1).clip(lower=1), axis=0)
    fig = go.Figure(go.Heatmap(z=norm.values, x=cm.columns, y=cm.index, text=cm.values, texttemplate="%{text}",
                               colorscale="Blues", zmin=0, zmax=1, colorbar=dict(title="recall")))
    fig.update_layout(height=700, xaxis_title="Predicted", yaxis_title="True", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    pc = pd.read_csv(config.TABLE_DIR / f"ml_per_class_{sel.replace(' ', '_').lower()}.csv")
    st.dataframe(pc.round(3), hide_index=True, width="stretch")
    st.subheader("Which inputs matter? (feature-group ablation)")
    st.dataframe(table("ml_ablation").round(3), hide_index=True, width="stretch")
    st.caption("Skills carry almost all the predictive signal; experience and location add little for role family.")
    st.subheader("Recommender ranking quality (held-out profiles)")
    st.dataframe(table("recommender_eval").round(3), hide_index=True, width="stretch")
    st.caption("The blended score is slightly below ML-only on top-k hit rate; it is kept because it also explains fit "
               "(experience, location, interest) and exposes skill gaps.")
    with st.expander("How data leakage was prevented"):
        st.markdown("- Label-like columns (`role`, `functional_area`, job title, description) are never features.\n"
                    "- Vocabulary, encoders and imputers are fitted inside a Pipeline on training rows only.\n"
                    "- Train/test split is grouped by company: **0 companies appear on both sides**.\n"
                    "- Models are tuned with grouped cross-validation on the training set; the test set is scored once.")
    with st.expander("Cross-validation grid (all settings tried)"):
        st.dataframe(table("ml_cv_results").round(3), hide_index=True, width="stretch")


def profile_form(key: str):
    rec = recommender()
    o = rec.options()
    with st.form(key):
        c1, c2 = st.columns(2)
        picked = c1.multiselect("Skills (pick from common ones)", o["skills"], default=[])
        extra = c1.text_input("More skills (comma separated)", placeholder="e.g. python, sql, tally, gst")
        years = c2.slider("Experience (years)", 0.0, 20.0, 1.0, 0.5)
        ug = c2.selectbox("Graduation degree", o["ug"])
        pg = c2.selectbox("Post-graduation", o["pg"], index=o["pg"].index("Post Graduation Not Required") if "Post Graduation Not Required" in o["pg"] else 0)
        city = c1.selectbox("Preferred city", ["(no preference)"] + o["cities"])
        ind = c2.selectbox("Interest / preferred industry", ["(no preference)"] + o["industries"])
        go_ = st.form_submit_button("Find my careers", type="primary")
    prof = {"skills": ", ".join(picked + ([extra] if extra else [])), "experience_years": years, "ug_degree": ug,
            "pg_degree": pg, "city": None if city.startswith("(") else city,
            "industry": None if ind.startswith("(") else ind}
    return go_, prof


def page_recommend():
    st.header("🎯 Career Recommendation")
    go_, prof = profile_form("rec_form")
    if go_:
        st.session_state["profile"] = prof
        st.session_state["result"] = recommender().recommend(prof, top_n=5)
    res = st.session_state.get("result")
    if not res:
        st.info("Fill in your profile and press **Find my careers**.")
        return
    for w in res["warnings"]:
        st.warning(w)
    g = build_guidance(res)
    st.markdown(render_markdown(g))
    st.subheader("Top recommended roles")
    st.caption("The match index is a *relative ranking* of roles for your profile (blend of model probability, skill coverage, "
               "experience, location and interest) - not a percentage chance. High-scoring profiles typically land well below 100.")
    for i, r in enumerate(res["recommendations"], 1):
        st.markdown(f"<div class='role-card'><b>{i}. {r['role']}</b> &nbsp; match index <b>{r['match_score']:.0f}/100</b> &nbsp;|&nbsp; model probability {100*r['components']['ml_probability']:.0f}%</div>", unsafe_allow_html=True)
        with st.expander("Why, gaps and roadmap", expanded=i == 1):
            a, b = st.columns([1, 1])
            with a:
                st.markdown("**Why this role**")
                for x in r["reasons"]:
                    st.write("- " + x)
                comp = pd.DataFrame({"component": list(r["components"]), "value": list(r["components"].values())})
                st.plotly_chart(bar(comp, "value", "component", "Score components (0-1)", ORANGE), width="stretch")
            with b:
                st.markdown("**Skill gaps**")
                if r["skill_gaps"]:
                    st.dataframe(pd.DataFrame(r["skill_gaps"]).rename(columns={"role_share_pct": "% of role postings asking"}), hide_index=True)
                else:
                    st.success("No major gaps among this role's most requested skills.")
                st.markdown("**Learning roadmap**")
                for p in r["roadmap"]:
                    st.write(f"**{p['phase']}:** {', '.join(p['focus'])}  \n_{p['why']}_")
    st.subheader("Ask about your results")
    q = st.text_input("e.g. Why this role? What should I learn first? What about salary or cities?")
    if q:
        st.info(answer_question(q, res))


def page_gap():
    st.header("🧩 Skill Gap Analysis")
    rec = recommender()
    roles = [str(c) for c in rec.classes]
    c1, c2 = st.columns([1, 2])
    role = c1.selectbox("Target role", roles, index=roles.index("Programming & Design") if "Programming & Design" in roles else 0)
    default = (st.session_state.get("profile") or {}).get("skills", "")
    skills = c2.text_input("Your skills (comma separated)", value=default, placeholder="e.g. java, sql, html")
    terms = rec.user_terms(skills)
    core = rec.core_skills[role].copy()
    core["status"] = ["You have it" if rec._covered(s, terms) else "Missing" for s in core["skill"]]
    core["share_pct"] = 100 * core["role_share"]
    cov = core.loc[core["status"] == "You have it", "role_share"].sum() / core["role_share"].sum()
    st.metric("Coverage of the role's in-demand skills (demand-weighted)", f"{100*cov:.0f}%")
    fig = px.bar(core, x="share_pct", y="skill", color="status", orientation="h",
                 color_discrete_map={"You have it": "#2e8b57", "Missing": "#c0392b"},
                 title=f"Most requested skills for {role}")
    fig.update_layout(height=560, yaxis=dict(autorange="reversed"), xaxis_title="% of this role's postings")
    st.plotly_chart(fig, width="stretch")
    prem = table("stats_skill_salary_premium")
    miss = core[core["status"] == "Missing"]["skill"]
    pm = prem[prem["skill"].isin(miss)][["skill", "exp_adjusted_premium_pct", "median_salary_with", "q_value_fdr"]]
    if len(pm):
        st.markdown("**Pay link of the missing skills** (experience-adjusted, descriptive only)")
        st.dataframe(pm.round(2), hide_index=True, width="stretch")


def page_salary():
    st.header("💰 Salary Prediction")
    sm = meta("salary_model_metadata.json")
    tm = sm["test_metrics"]
    if not sm["reliable"]:
        st.error("The salary model did not pass the reliability checks, so predictions are disabled.")
        st.json(sm["checks"])
        return
    st.success("Reliability checks passed on held-out companies: " + "; ".join(sm["checks"]))
    a, b, c, d = st.columns(4)
    a.metric("Test R² (log salary)", f"{tm['r2_log']:.2f}")
    b.metric("Mean abs. error", f"{tm['mae_lakh']:.2f} lakh")
    c.metric("Gain vs global median", f"{sm['mae_gain_vs_global_median_pct']:.0f}%")
    d.metric("Gain vs experience-only", f"{sm['mae_gain_vs_experience_only_pct']:.0f}%")
    st.caption(f"The model beats a plain experience-only formula by only {sm['mae_gain_vs_experience_only_pct']:.0f}% in MAE: "
               "most of the signal is experience. 80% intervals covered "
               f"{100*sm['interval_test_coverage']:.0f}% of test postings.")
    st.subheader("Estimate for a profile")
    go_, prof = profile_form("sal_form")
    rec = recommender()
    roles = sorted(sm["role_salary_support"], key=lambda r: -sm["role_salary_support"][r])
    role = st.selectbox("Role", roles, format_func=lambda r: f"{r} ({sm['role_salary_support'][r]} salaried postings)")
    if go_:
        out = rec.predict_salary(prof, role)
        x, y, z = st.columns(3)
        x.metric("Estimate", f"{out['estimate_lpa']:.1f} lakh / year")
        y.metric("80% range low", f"{out['low_lpa']:.1f}")
        z.metric("80% range high", f"{out['high_lpa']:.1f}")
        if out["low_support"]:
            st.warning("Few salaried postings exist for this role; treat the estimate as rough.")
    pr = pd.read_csv(config.TABLE_DIR / "salary_test_predictions.csv")
    fig = px.scatter(pr, x="actual_lpa", y="predicted_lpa", opacity=0.5, color_discrete_sequence=[BLUE],
                     title="Held-out test: predicted vs actual advertised salary (lakh)")
    mx = float(max(pr.actual_lpa.max(), pr.predicted_lpa.max()))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=ORANGE, dash="dash"), name="perfect"))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Caveats"):
        for c_ in sm["caveats"]:
            st.write("- " + c_)
        st.dataframe(table("salary_test_metrics").round(3).drop(columns=["params"]), hide_index=True)


PAGES = {"🏠 Overview / SDG 8": page_overview, "📊 Job Market Analytics": page_analytics,
         "📐 Statistical Analysis": page_stats, "🤖 ML Performance": page_ml,
         "🎯 Career Recommendation": page_recommend, "🧩 Skill Gap Analysis": page_gap,
         "💰 Salary Prediction": page_salary}

st.sidebar.title("🧭 Career Intelligence")
choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
if not need_artifacts():
    _k = kpis()
    st.sidebar.caption(f"**Data:** {_k['total_postings']:,} Naukri.com postings (2015-2016)  \n"
                       f"**Pay disclosed:** {_k['salary_disclosed_pct']:.0f}% of postings")
st.sidebar.caption("SDG 8 | Python, scikit-learn, SciPy, Streamlit")
if need_artifacts():
    st.error("Model files not found. Run `python run_pipeline.py` first.")
else:
    PAGES[choice]()
