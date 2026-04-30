# ============================================================
# data_loader.py — Download & cache historical price data
# ============================================================

import os
import time
import warnings
import pandas as pd
import yfinance as yf
from config import UNIVERSE, BENCHMARK_TICKER, BENCHMARK_FALLBACK, START_DATE, TEST_END_DATE, DATA_DIR, MIN_HISTORY_DATE

warnings.filterwarnings("ignore")


def download_all(force_refresh=False):
    """
    Download OHLCV data for all stocks + benchmark.
    Saves each ticker as a parquet file in DATA_DIR.
    Returns a dict: {ticker: DataFrame}
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    all_data = {}
    tickers = UNIVERSE + [BENCHMARK_TICKER, BENCHMARK_FALLBACK]

    print(f"\n📥 Downloading data for {len(tickers)} tickers...")
    print(f"   Period: {START_DATE} → {TEST_END_DATE}\n")

    for i, ticker in enumerate(tickers, 1):
        safe_name = ticker.replace("^", "IDX_").replace(".", "_")
        filepath = os.path.join(DATA_DIR, f"{safe_name}.parquet")

        if os.path.exists(filepath) and not force_refresh:
            df = pd.read_parquet(filepath)
            all_data[ticker] = df
            print(f"  [{i:02d}/{len(tickers)}] {ticker:<20} ✓ loaded from cache ({len(df)} rows)")
            continue

        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=TEST_END_DATE,
                auto_adjust=True,
                progress=False,
            )

            if df.empty or len(df) < 100:
                print(f"  [{i:02d}/{len(tickers)}] {ticker:<20} ✗ insufficient data, skipping")
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Remove rows where close is 0 or NaN
            df = df[df["Close"] > 0].dropna(subset=["Close"])

            df.to_parquet(filepath)
            all_data[ticker] = df
            print(f"  [{i:02d}/{len(tickers)}] {ticker:<20} ✓ downloaded ({len(df)} rows)")
            time.sleep(0.3)  # Avoid rate limiting

        except Exception as e:
            print(f"  [{i:02d}/{len(tickers)}] {ticker:<20} ✗ error: {e}")

    print(f"\n✅ Data ready for {len(all_data)} tickers\n")
    return all_data


def get_price_panel(all_data):
    """
    Build a clean closing price panel: DataFrame [dates x tickers]
    Only includes stocks that have data from at least 2016 onward.
    """
    close_dict = {}
    for ticker, df in all_data.items():
        if ticker in [BENCHMARK_TICKER, BENCHMARK_FALLBACK]:
            continue
        if df.index.min() <= pd.Timestamp(MIN_HISTORY_DATE):
            close_dict[ticker] = df["Close"]

    panel = pd.DataFrame(close_dict)
    panel.index = pd.to_datetime(panel.index)
    panel.sort_index(inplace=True)

    # Forward-fill up to 5 days (handles trading holidays)
    panel = panel.ffill(limit=5)

    print(f"📊 Price panel shape: {panel.shape}  ({panel.shape[1]} stocks × {panel.shape[0]} days)")
    return panel


def get_volume_panel(all_data):
    """Build a volume panel: DataFrame [dates x tickers]"""
    vol_dict = {}
    for ticker, df in all_data.items():
        if ticker in [BENCHMARK_TICKER, BENCHMARK_FALLBACK]:
            continue
        if df.index.min() <= pd.Timestamp(MIN_HISTORY_DATE):
            vol_dict[ticker] = df["Volume"]

    panel = pd.DataFrame(vol_dict)
    panel.index = pd.to_datetime(panel.index)
    panel.sort_index(inplace=True)
    panel = panel.ffill(limit=5).fillna(0)
    return panel


def get_benchmark(all_data):
    """Return benchmark close price series. Tries Nifty 500 first, falls back to Nifty 50."""
    if BENCHMARK_TICKER in all_data and not all_data[BENCHMARK_TICKER].empty:
        df = all_data[BENCHMARK_TICKER]
        print(f"   Using benchmark: Nifty 500 ({BENCHMARK_TICKER})")
        return df["Close"]
    elif BENCHMARK_FALLBACK in all_data and not all_data[BENCHMARK_FALLBACK].empty:
        df = all_data[BENCHMARK_FALLBACK]
        print(f"   Using benchmark: Nifty 50 fallback ({BENCHMARK_FALLBACK})")
        return df["Close"]
    raise ValueError("No benchmark data found. Re-run with force_refresh=True.")
