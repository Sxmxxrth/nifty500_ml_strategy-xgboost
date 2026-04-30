# ============================================================
# features.py — Signal engineering (no data leakage!)
# ============================================================
# ⚠️  KEY RULE: Every feature at date T uses ONLY data up to T.
#     Forward-looking target is created SEPARATELY and never
#     mixed into the feature computation.
# ============================================================

import numpy as np
import pandas as pd
from config import FORWARD_DAYS


# ── Helpers ──────────────────────────────────────────────────

def _pct_return(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """n-day percentage return. Safe: uses only past data."""
    return close.pct_change(days)


def _rolling_std(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation of daily returns."""
    return close.pct_change().rolling(window, min_periods=int(window * 0.7)).std()


def _rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI (Relative Strength Index) — vectorised across all stocks."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _volume_ratio(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Current volume / rolling average volume."""
    avg_vol = volume.rolling(window, min_periods=int(window * 0.7)).mean()
    return volume / (avg_vol + 1)


def _distance_from_high(close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """How far (%) is current price from the rolling high."""
    rolling_high = close.rolling(window, min_periods=int(window * 0.5)).max()
    return (close - rolling_high) / (rolling_high + 1e-9)


def _relative_strength(close: pd.DataFrame, benchmark: pd.Series, days: int) -> pd.DataFrame:
    """Stock return minus benchmark return over `days` period."""
    stock_ret = close.pct_change(days)
    bench_ret = benchmark.pct_change(days)
    return stock_ret.subtract(bench_ret, axis=0)


# ── Main feature builder ──────────────────────────────────────

def build_features(close: pd.DataFrame,
                   volume: pd.DataFrame,
                   benchmark: pd.Series) -> pd.DataFrame:
    """
    Compute all features for all stocks on all dates.
    Returns a tall DataFrame: index = (date, ticker)

    Features:
    - Momentum      : 1m, 3m, 6m, 12m returns
    - Mean Reversion: RSI(14), distance from 52-week high
    - Volatility    : 20d, 60d rolling std of returns
    - Volume        : 20d volume ratio
    - Rel. Strength : stock vs benchmark (1m, 3m)
    """
    print("⚙️  Engineering features...")

    feature_frames = {}

    # --- Momentum ---
    feature_frames["mom_1m"]  = _pct_return(close, 21)
    feature_frames["mom_3m"]  = _pct_return(close, 63)
    feature_frames["mom_6m"]  = _pct_return(close, 126)
    feature_frames["mom_12m"] = _pct_return(close, 252)

    # --- Mean Reversion ---
    feature_frames["rsi_14"]          = _rsi(close, 14)
    feature_frames["dist_52w_high"]   = _distance_from_high(close, 252)

    # --- Volatility ---
    feature_frames["vol_20d"] = _rolling_std(close, 20)
    feature_frames["vol_60d"] = _rolling_std(close, 60)

    # --- Volume ---
    feature_frames["vol_ratio_20d"] = _volume_ratio(volume, 20)

    # --- Relative Strength vs Nifty ---
    feature_frames["rel_str_1m"] = _relative_strength(close, benchmark, 21)
    feature_frames["rel_str_3m"] = _relative_strength(close, benchmark, 63)

    print(f"   Built {len(feature_frames)} features across {close.shape[1]} stocks")

    # Stack into a MultiIndex (date, ticker) DataFrame
    stacked_frames = []
    for name, df in feature_frames.items():
        s = df.stack()
        s.name = name
        stacked_frames.append(s)

    features_df = pd.concat(stacked_frames, axis=1)
    features_df.index.names = ["date", "ticker"]

    print(f"   Feature matrix shape: {features_df.shape}")
    return features_df


# ── Target variable ───────────────────────────────────────────

def build_target(close: pd.DataFrame) -> pd.Series:
    """
    Binary target: 1 if stock's FORWARD return > median cross-sectional
    return that period, else 0.

    ⚠️  IMPORTANT: Target uses future data (shift(-FORWARD_DAYS)).
    This is fine — we never let model training see future features,
    only this future return as the label.
    """
    # Forward return for each stock
    fwd_return = close.shift(-FORWARD_DAYS) / close - 1

    # Cross-sectional median on each date
    median_return = fwd_return.median(axis=1)

    # Binary: 1 if above median, 0 otherwise
    target = fwd_return.gt(median_return, axis=0).astype(int)

    # Stack to (date, ticker)
    target = target.stack()
    target.index.names = ["date", "ticker"]
    target.name = "target"

    return target


# ── Combine features + target ─────────────────────────────────

def build_dataset(close: pd.DataFrame,
                  volume: pd.DataFrame,
                  benchmark: pd.Series) -> pd.DataFrame:
    """
    Merges features and target into one training-ready DataFrame.
    Drops rows with NaN features or target.
    """
    features = build_features(close, volume, benchmark)
    target   = build_target(close)

    dataset = features.join(target, how="inner")
    dataset.dropna(inplace=True)

    # Sanity check: target should be roughly 50/50
    target_pct = dataset["target"].mean()
    print(f"   Target balance: {target_pct:.1%} positive (ideally ~50%)")

    return dataset


FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "rsi_14", "dist_52w_high",
    "vol_20d", "vol_60d",
    "vol_ratio_20d",
    "rel_str_1m", "rel_str_3m",
]
