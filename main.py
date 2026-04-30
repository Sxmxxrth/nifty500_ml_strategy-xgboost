# ============================================================
# main.py — Run the full strategy end to end
# ============================================================
# Usage:
#   python main.py              → Full pipeline (download + train + backtest)
#   python main.py --skip-dl   → Skip download if data already cached
# ============================================================

import os
import sys
import time
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# ── Parse arguments ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-dl",     action="store_true", help="Skip data download (use cache)")
parser.add_argument("--skip-train",  action="store_true", help="Skip training (load saved model)")
parser.add_argument("--n-iter",      type=int, default=20, help="RandomizedSearchCV iterations (default 20)")
args = parser.parse_args()

from config     import OUTPUT_DIR, DATA_DIR
from data_loader import download_all, get_price_panel, get_volume_panel, get_benchmark
from features   import build_dataset, build_features, FEATURE_COLS
from model      import get_train_test_split, train_model, evaluate_model
from backtest   import get_rebalance_dates, run_backtest, run_benchmark_nav
from metrics    import compute_metrics, print_metrics
from visualize  import (plot_nav, plot_metrics_table,
                        plot_monthly_heatmap, plot_feature_importance,
                        plot_rolling_sharpe)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

MODEL_PATH   = os.path.join(OUTPUT_DIR, "model.pkl")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.parquet")


def main():
    t_start = time.time()

    print("=" * 60)
    print("  🚀 NIFTY ML STRATEGY — FULL PIPELINE")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # STEP 1: Data Download
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 1] Data Preparation")
    print("-" * 40)

    all_data      = download_all(force_refresh=not args.skip_dl)
    price_panel   = get_price_panel(all_data)
    volume_panel  = get_volume_panel(all_data)
    benchmark     = get_benchmark(all_data)

    if price_panel.empty:
        print("❌ No price data available. Check your internet connection.")
        sys.exit(1)

    print(f"\n  Stocks loaded : {price_panel.shape[1]}")
    print(f"  Date range    : {price_panel.index.min().date()} → {price_panel.index.max().date()}")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Feature Engineering
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 2] Feature Engineering")
    print("-" * 40)

    if os.path.exists(DATASET_PATH) and args.skip_dl:
        print("  Loading cached dataset...")
        dataset = pd.read_parquet(DATASET_PATH)
        # Also build the full features_df for scoring
        features_df = build_features(price_panel, volume_panel, benchmark)
    else:
        from features import build_dataset
        dataset     = build_dataset(price_panel, volume_panel, benchmark)
        features_df = build_features(price_panel, volume_panel, benchmark)
        dataset.to_parquet(DATASET_PATH)
        print(f"  Dataset saved → {DATASET_PATH}")

    print(f"\n  Total rows    : {len(dataset):,}")
    print(f"  Features      : {FEATURE_COLS}")

    # ──────────────────────────────────────────────────────────
    # STEP 3: Train / Load Model
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 3] Model Training")
    print("-" * 40)

    train_df, test_df = get_train_test_split(dataset)

    if args.skip_train and os.path.exists(MODEL_PATH):
        print("  Loading saved model...")
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        print("  ✓ Model loaded")
    else:
        pipeline = train_model(train_df, n_iter=args.n_iter)

        # Save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"  Model saved → {MODEL_PATH}")

    # Out-of-sample model evaluation
    model_metrics = evaluate_model(pipeline, test_df)

    # ──────────────────────────────────────────────────────────
    # STEP 4: Backtesting
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 4] Backtesting")
    print("-" * 40)

    rebalance_dates = get_rebalance_dates(price_panel)

    results = run_backtest(
        pipeline       = pipeline,
        features_df    = features_df,
        price_panel    = price_panel,
        rebalance_dates= rebalance_dates,
    )

    nav           = results["nav"]
    trades_df     = results["trades"]
    benchmark_nav = run_benchmark_nav(benchmark)

    # Save NAV and trades
    nav.to_csv(os.path.join(OUTPUT_DIR, "nav.csv"), header=["NAV"])
    trades_df.to_csv(os.path.join(OUTPUT_DIR, "trades.csv"), index=False)
    print(f"\n  NAV saved    → {OUTPUT_DIR}/nav.csv")
    print(f"  Trades saved → {OUTPUT_DIR}/trades.csv")

    # ──────────────────────────────────────────────────────────
    # STEP 5: Performance Analysis
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 5] Performance Metrics")
    print("-" * 40)

    metrics = compute_metrics(nav, benchmark_nav)
    print_metrics(metrics)

    # Save metrics to text
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("NIFTY ML STRATEGY — PERFORMANCE METRICS\n")
        f.write("=" * 50 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k:<30}: {v}\n")
    print(f"  Metrics saved → {metrics_path}")

    # ──────────────────────────────────────────────────────────
    # STEP 6: Visualizations
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 6] Generating Charts")
    print("-" * 40)

    plot_nav(nav, benchmark_nav, metrics)
    plot_metrics_table(metrics)
    plot_monthly_heatmap(nav)
    plot_feature_importance(pipeline)
    plot_rolling_sharpe(nav)

    # ──────────────────────────────────────────────────────────
    # STEP 7: Summary
    # ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  ⏱️  Total time  : {elapsed/60:.1f} minutes")
    print(f"  📁 Output folder: {OUTPUT_DIR}/")
    print()
    print("  Files generated:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
        print(f"    • {fname:<30} ({size/1024:.1f} KB)")

    print()
    print("  📊 Key Results:")
    print(f"    Strategy CAGR   : {metrics.get('CAGR (%)', 'N/A')}%")
    print(f"    Benchmark CAGR  : {metrics.get('Benchmark CAGR (%)', 'N/A')}%")
    print(f"    Sharpe Ratio    : {metrics.get('Sharpe Ratio', 'N/A')}")
    print(f"    Max Drawdown    : {metrics.get('Max Drawdown (%)', 'N/A')}%")
    print(f"    Alpha (ann.)    : {metrics.get('Alpha (ann. %)', 'N/A')}%")
    print()


if __name__ == "__main__":
    main()
