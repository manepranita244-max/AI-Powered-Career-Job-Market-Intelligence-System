AI-Powered Career & Job Market Intelligence System

**SDG 8 - Decent Work and Economic Growth** | one project for the *AICTE / IBM SkillsBuild Machine Learning & Applied AI internship* and a *college Statistics + Machine Learning mini-project*.

The system analyses a real job-market dataset, predicts suitable job-role families from a person's skills, education, experience, location and interests, recommends careers with match scores, skill gaps and a learning roadmap, and explains everything in a Streamlit dashboard. **Every number in this README was computed by the code from the dataset** (the tables below are copied from `reports/tables/`).

1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py            # full pipeline, ~10-12 min on 1 CPU   (use --quick for ~1 min: data+EDA+stats)
streamlit run app/streamlit_app.py
pytest -q                         # 12 tests
```
The pre-built `models/` and `reports/` folders are included, so you can run the dashboard immediately. Keep scikit-learn at **1.8.0** and NumPy in the **2.4.x** series to load the included serialized models; `requirements.txt` pins these compatible versions. The raw CSV is included; if it is missing, `src/preprocessing.py` downloads it from GitHub and verifies its SHA-256 checksum. Developed on Python 3.12 (code uses nothing newer than 3.11 features, but 3.11 itself was **not** tested here).

2. Dataset

* **Source:** Naukri.com job postings (public CSV `naukri_com-job_sample.csv`, obtained from the GitHub repo `DebrajAdhikary5/naukri_com_job_analysis`). Check the original licence/terms before publishing derived work.
* **Size:** 21,936 raw rows -> 10,938 blank rows and 33 duplicates removed -> **10,965 clean postings** from 4,925 employers ({'2015.0': 1635, '2016.0': 9310}).
* Role category, role, functional area and key skills are parsed out of the free-text description; experience, pay, education and cities are parsed with unit-tested rules (`src/preprocessing.py`).
* Salary is disclosed in only **3,186 postings (29.06%)**.

**Limitations (please state these in your report/viva):** postings are from 2015-2016, so demand and salaries are historical; pay disclosure is not random; 82% of postings are in three cities, so location signal is weak elsewhere; skills are free text, so a "skill" here is a 1-2 word term, not a curated taxonomy.

3. Project structure

```
data/raw/            original CSV            data/processed/   cleaned dataset
src/preprocessing.py cleaning + parsers      src/skills.py     skill terms, SkillIndex (demand/lift per role)
src/eda.py           EDA tables + figures    src/statistics_analysis.py   statistics + hypothesis tests
src/features.py      leakage-safe features   src/train_models.py          4 models, CV, evaluation, saving
src/salary_model.py  salary model + checks   src/recommender.py           match score, gaps, roadmap, evaluation
src/guidance.py      grounded guidance       app/streamlit_app.py         7-page dashboard
models/              joblib models + JSON metadata       reports/{tables,figures}   all outputs
tests/               pytest suite            run_pipeline.py              one command to rebuild everything
```

4. Module results

4.1 EDA
10,965 postings, 82 role categories, 62 industries. 15.4% of postings accept freshers (0 years minimum); the largest industry (*IT-Software / Software Services*) holds 40.8% of postings; median advertised salary is 3.50 lakh/year. Eight figures are in `reports/figures/`; the dashboard shows interactive versions.

4.2 Statistics
Descriptive statistics (mean, median, mode, variance, SD, quartiles, IQR, skewness, kurtosis), Pearson/Spearman correlation, IQR and z-score outliers, normality tests, and seven hypothesis tests. Salary is strongly right-skewed and non-normal, so rank-based tests are used; p-values are Holm-adjusted and each test reports an effect size. Key hand-checks: Spearman, Kruskal-Wallis H and chi-square statistics were recomputed manually and matched exactly.

| id | question | p | eff | dec |
|---|---|---|---|---|
| H1 | Do salaries differ across job role categories? | <0.001 | epsilon_sq=0.238 (large) | Reject H0 |
| H2 | Do IT-Software postings pay differently from other industries? | <0.001 | cliffs_delta=0.325 (small) | Reject H0 |
| H3 | Is higher required experience associated with higher salary? | <0.001 | spearman_rho=0.724 (large) | Reject H0 |
| H4 | Do salaries differ between the main cities? | <0.001 | epsilon_sq=0.007 (negligible) | Reject H0 |
| H5 | Does the mix of job roles depend on the city? | <0.001 | cramers_v=0.145 (small) | Reject H0 |
| H6 | Are fresher-friendly postings more (or less) likely to disclose pay? (SDG 8 transparency) | <0.001 | cohens_h=0.776 (medium) | Reject H0 |
| H7 | Do Programming & Design roles ask for more experience than Voice (BPO) roles? | <0.001 | cliffs_delta=0.683 (large) | Reject H0 |

Statistically significant does not mean large: e.g. H4 (city) is significant with a *negligible* effect, because cities differ in role/experience mix.

4.3 Machine learning - job-role prediction
Target: **24 role families** (`Role Category`, "Other" excluded, >= 60 postings), 6,798 postings with listed skills. Features: skills, experience, education (UG/PG), location, industry interest.

**Leakage prevention:** `role`, `functional_area`, job title and description are never features; vocabulary/encoders/imputers are fitted inside a scikit-learn `Pipeline` on training rows only; the train/test split is **grouped by company** (0 shared companies) and stratified by role; tuning uses grouped cross-validation on the training set; the test set is scored once.

Held-out test results (1,360 postings from unseen companies):

| model | accuracy | precision_macro | recall_macro | f1_macro | top3_accuracy |
|---|---|---|---|---|---|
| Baseline: most frequent | 0.384 | 0.016 | 0.042 | 0.023 | 0.426 |
| Baseline: stratified | 0.190 | 0.044 | 0.045 | 0.044 | 0.235 |
| Decision Tree | 0.584 | 0.342 | 0.335 | 0.334 | 0.654 |
| Gradient Boosting | 0.646 | 0.460 | 0.486 | 0.457 | 0.884 |
| Logistic Regression | 0.664 | 0.458 | 0.510 | 0.475 | 0.886 |
| Random Forest | 0.659 | 0.487 | 0.537 | 0.485 | 0.897 |

Selected model: **Logistic Regression** (best cross-validated macro-F1, chosen *before* looking at test scores). On test, Random Forest has a slightly higher macro-F1; that is reported, not hidden. 95% company-bootstrap interval for the selected model: accuracy 0.624-0.704, macro-F1 0.438-0.509. Macro metrics are lower than accuracy because small role families are hard; main confusions are between the three Sales categories (see `reports/figures/09_confusion_matrix_best.png`).

Which inputs matter (drop one group, refit, test):

| removed_group | accuracy | f1_macro | f1_macro_change |
|---|---|---|---|
| (none - full model) | 0.664 | 0.475 | 0.000 |
| skills | 0.354 | 0.263 | -0.212 |
| experience | 0.662 | 0.476 | 0.001 |
| education | 0.661 | 0.466 | -0.008 |
| location | 0.661 | 0.474 | -0.000 |
| interest (industry) | 0.647 | 0.456 | -0.018 |

Skills carry nearly all the signal for *role family*; experience and location matter for fit and pay rather than for the role label.

4.4 Salary prediction (included because it passed reliability checks)
Target: log of the advertised salary-range midpoint (2,984 salaried postings with skills). Reliability gate (`src/config.py`): >= 1,000 training rows, test R2(log) >= 0.5, MAE >= 25% better than a global-median baseline - **all passed**.

| model | r2_log | mae_lakh | median_abs_pct_error | within_30pct_share |
|---|---|---|---|---|
| Baseline: global median | -0.045 | 3.234 | 46.2 | 34.0 |
| Baseline: experience only (quadratic) | 0.595 | 2.105 | 29.1 | 50.8 |
| Gradient Boosting | 0.694 | 1.857 | 25.2 | 58.8 |
| Random Forest | 0.691 | 1.858 | 22.4 | 62.5 |
| Ridge | 0.657 | 2.012 | 25.7 | 55.5 |

Selected: Gradient Boosting. It beats the global median by 43% in MAE but the plain experience-only formula by just **12%** - most of the signal is experience. 80% prediction intervals (x0.60 to x1.69) covered 82.8% of test postings. It predicts the *typical advertised* salary, not an individual's pay.

4.5 Career recommendation
`match index = 0.55*ML probability + 0.25*skill coverage + 0.10*experience fit + 0.05*location fit + 0.05*interest fit` (weights are a declared design choice, not tuned on test data). Output per role: score, reasons, matched skills, prioritised skill gaps (from each role's most requested distinctive skills) and a 3-phase roadmap. The index is a **relative ranking**, not a probability; typical scores are well below 100.

Offline check on held-out postings (does the true role appear in the top-k?):

| ranking | top1 | top3 | top5 | mean_reciprocal_rank |
|---|---|---|---|---|
| ML probability only | 0.664 | 0.886 | 0.946 | 0.786 |
| Skill coverage only | 0.466 | 0.637 | 0.726 | 0.583 |
| Blend (default weights) | 0.660 | 0.870 | 0.913 | 0.776 |
| Blend without location/interest | 0.663 | 0.871 | 0.924 | 0.780 |

Honest finding: the blend is slightly *below* ML-only on top-k hit rate. It is kept because it also explains fit and exposes skill gaps; switch weights in `src/recommender.py` (`WEIGHTS`) if you prefer pure ML ranking.

4.6 Applied AI guidance
`src/guidance.py` assembles advice **only from computed results** (scores, gaps, experience ranges, salary estimates, evaluation numbers) and answers follow-up questions by keyword routing over the same results. It is deliberately not a free-text chatbot: it cannot invent a fact, and a sentence is skipped when its number is unavailable.

4.7 Dashboard
Seven pages: Overview / SDG 8, Job Market Analytics, Statistical Analysis, ML Performance, Career Recommendation, Skill Gap Analysis, Salary Prediction. All pages and the recommendation, Q&A and salary forms were exercised with Streamlit's `AppTest` without errors.

5. Using this for your submission
* **Internship:** emphasise the applied pipeline (data -> model -> recommender -> guidance -> dashboard), leakage controls and the SDG-8 indicators on the Overview page.
* **Statistics + ML mini-project:** emphasise `src/statistics_analysis.py` (assumption checks, effect sizes, multiple-testing correction) and the model comparison with grouped CV.
* Be ready to explain: why grouped splits, why macro-F1, why Spearman/Kruskal instead of Pearson/ANOVA, why the salary result is "typical advertised pay", and why the blend does not beat ML-only.

6. Troubleshooting
* *"Model files not found"* in the app -> run `python run_pipeline.py`.
* Training is single-threaded friendly; full run time depends on your CPU.
* Library-version errors -> check `requirements.txt` (minimum versions and the features they enable).
* If the download is blocked, place the CSV at `data/raw/naukri_com-job_sample.csv` manually.

7. Ideas for extension
Fresher data from a current source, curated skill taxonomy, role-level (not family-level) prediction, calibrated probabilities, an optional LLM layer that only rephrases the computed facts.
