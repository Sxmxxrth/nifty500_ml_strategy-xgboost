# ============================================================
# backtest.py — Portfolio simulation engine
# ============================================================
# Logic:
#   Every 4th Thursday → score all stocks → pick Top N →
#   rebalance portfolio → apply transaction costs → track NAV
# ============================================================

import numpy as np
import pandas as pd
from config import (
    TOP_N_STOCKS, TRANSACTION_COST,
    TEST_START_DATE, TEST_END_DATE, REBALANCE_FREQ
)


def get_rebalance_dates(price_panel: pd.DataFrame) -> list:
    """
    Generate rebalance dates (every 4th Thursday) within test period.
    Snaps to nearest available trading day if needed.
    """
    date_range = pd.date_range(
        start=TEST_START_DATE,
        end=TEST_END_DATE,
        freq=REBALANCE_FREQ,       # Every 4 weeks on Thursday
    )

    trading_days = price_panel.index

    rebalance_dates = []
    for d in date_range:
        # Find nearest available trading day <= d
        available = trading_days[trading_days <= d]
        if len(available) > 0:
            rebalance_dates.append(available[-1])

    # Deduplicate
    rebalance_dates = sorted(set(rebalance_dates))
    print(f"\n📅 Rebalance dates: {len(rebalance_dates)} periods "
          f"({rebalance_dates[0].date()} → {rebalance_dates[-1].date()})")
    return rebalance_dates


def run_backtest(pipeline,
                 features_df: pd.DataFrame,
                 price_panel: pd.DataFrame,
                 rebalance_dates: list) -> dict:
    """
    Full portfolio simulation.

    At each rebalance date:
      1. Score all stocks using the trained model
      2. Select Top N stocks
      3. Calculate turnover vs previous portfolio
      4. Apply transaction costs on turnover
      5. Track daily NAV between rebalances

    Returns a dict with:
      - 'nav'        : daily portfolio NAV (starts at 100)
      - 'holdings'   : dict of {date: [tickers held]}
      - 'trade_log'  : list of trade details per rebalance
    """
    from model import score_stocks_on_date

    print(f"\n🔄 Running backtest — Top {TOP_N_STOCKS} stocks, "
          f"Transaction cost: {TRANSACTION_COST*10000:.0f} bps\n")

    nav_series     = {}   # date → portfolio value
    holdings_log   = {}   # rebalance date → list of tickers
    trade_log      = []   # trade details

    portfolio_value = 100.0          # Start with 100
    current_holdings = {}            # {ticker: weight}
    current_prices   = {}            # {ticker: entry price}

    # Build a mapping of trading days for fast lookup
    trading_days = sorted(price_panel.index)

    def get_price(ticker, date):
        """Get closing price for ticker on date (or nearest prior day)."""
        available = price_panel.index[price_panel.index <= date]
        if len(available) == 0 or ticker not in price_panel.columns:
            return None
        return price_panel.loc[available[-1], ticker]

    # ── Main rebalance loop ──────────────────────────────────
    for i, reb_date in enumerate(rebalance_dates):

        # --- Step 1: Score all stocks on this date ---
        scores = score_stocks_on_date(pipeline, features_df, reb_date)

        if len(scores) < TOP_N_STOCKS:
            print(f"  [{i+1:02d}] {reb_date.date()} — Not enough stocks scored ({len(scores)}), skipping")
            continue

        # --- Step 2: Pick Top N ---
        top_stocks = scores.nlargest(TOP_N_STOCKS).index.tolist()
        new_weight  = 1.0 / TOP_N_STOCKS   # Equal weight

        # --- Step 3: Calculate turnover ---
        old_tickers = set(current_holdings.keys())
        new_tickers = set(top_stocks)

        sold    = old_tickers - new_tickers   # Exiting positions
        bought  = new_tickers - old_tickers   # Entering positions
        kept    = old_tickers & new_tickers   # Unchanged

        turnover = (len(sold) + len(bought)) / max(len(new_tickers), 1)

        # --- Step 4: Apply transaction costs ---
        cost = turnover * TRANSACTION_COST * portfolio_value
        portfolio_value -= cost

        # --- Step 5: Update holdings ---
        current_holdings = {ticker: new_weight for ticker in top_stocks}
        current_prices   = {
            ticker: get_price(ticker, reb_date)
            for ticker in top_stocks
        }

        holdings_log[reb_date] = top_stocks

        trade_log.append({
            "date":            reb_date,
            "portfolio_value": portfolio_value,
            "top_stocks":      top_stocks,
            "stocks_sold":     list(sold),
            "stocks_bought":   list(bought),
            "turnover":        turnover,
            "cost":            cost,
            "n_scored":        len(scores),
        })

        print(f"  [{i+1:02d}] {reb_date.date()} | "
              f"NAV: {portfolio_value:7.2f} | "
              f"Bought: {len(bought):2d}  Sold: {len(sold):2d}  Kept: {len(kept):2d} | "
              f"Cost: ₹{cost:.2f}")

        # --- Step 6: Track daily NAV until next rebalance ---
        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else pd.Timestamp(TEST_END_DATE)

        # Get trading days in this holding period
        period_days = [d for d in trading_days if reb_date < d <= next_reb]

        prev_day = reb_date
        for day in period_days:
            daily_return = 0.0
            valid_count  = 0

            for ticker, weight in current_holdings.items():
                p0 = get_price(ticker, prev_day)
                p1 = get_price(ticker, day)

                if p0 is not None and p1 is not None and p0 > 0:
                    daily_return += weight * (p1 / p0 - 1)
                    valid_count  += 1

            # Scale by valid stocks
            if valid_count > 0:
                daily_return = daily_return * (len(current_holdings) / valid_count)

            portfolio_value *= (1 + daily_return)
            nav_series[day]  = portfolio_value
            prev_day         = day

    # ── Set initial NAV point ────────────────────────────────
    first_date = rebalance_dates[0] if rebalance_dates else pd.Timestamp(TEST_START_DATE)
    nav_series[first_date] = 100.0

    nav = pd.Series(nav_series).sort_index()

    print(f"\n✅ Backtest complete!")
    print(f"   Final NAV     : {nav.iloc[-1]:.2f}")
    print(f"   Total return  : {(nav.iloc[-1]/100 - 1)*100:.1f}%")
    print(f"   Rebalances    : {len(trade_log)}")

    return {
        "nav":      nav,
        "holdings": holdings_log,
        "trades":   pd.DataFrame(trade_log),
    }


def run_benchmark_nav(benchmark: pd.Series) -> pd.Series:
    """
    Normalise benchmark (Nifty 50) to start at 100 on TEST_START_DATE.
    """
    bench = benchmark.copy()
    bench = bench[bench.index >= TEST_START_DATE]
    bench = bench[bench.index <= TEST_END_DATE]
    bench = bench.dropna()

    if bench.empty:
        return bench

    bench_nav = bench / bench.iloc[0] * 100
    bench_nav.name = "Nifty50"
    return bench_nav
