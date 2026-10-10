"""Run the whole project end to end.

    python run_pipeline.py            # everything (about 10-12 minutes on 1 CPU)
    python run_pipeline.py --quick    # data + EDA + statistics only (about 1 minute)

Steps: clean data -> EDA -> statistics -> role models -> salary model -> recommender check
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")


def step(name, fn):
    t = time.time()
    print(f"\n=== {name} ===", flush=True)
    out = fn()
    print(f"--- {name} finished in {time.time() - t:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip model training")
    args = ap.parse_args()

    from src import eda, preprocessing, statistics_analysis

    df = step("1. Clean data", lambda: preprocessing.build_clean_dataset(verbose=False))
    print(f"clean rows: {len(df):,}")
    step("2. Exploratory data analysis", lambda: eda.run_eda(df))
    step("3. Statistical analysis", lambda: statistics_analysis.run_statistics(df))
    if not args.quick:
        from src import recommender, salary_model, train_models
        step("4. Role prediction models", train_models.main)
        step("5. Salary model", salary_model.main)
        step("6. Recommender evaluation", recommender.evaluate)
    print("\nDone. Launch the dashboard with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
