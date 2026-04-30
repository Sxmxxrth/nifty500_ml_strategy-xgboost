# ============================================================
# metrics.py — Performance analytics
# ============================================================

import numpy as np
import pandas as pd


def compute_metrics(nav: pd.Series, benchmark_nav: pd.Series = None) -> dict:
    """
    Compute standard quant performance metrics from a NAV series.

    Parameters
    ----------
    nav           : Portfolio NAV series (starts at 100)
    benchmark_nav : Optional benchmark NAV series (same start)

    Returns
    -------
    dict of metrics
    """
    nav = nav.dropna().sort_index()

    # Daily returns
    daily_returns = nav.pct_change().dropna()

    # ── Core return metrics ──────────────────────────────────
    total_days  = (nav.index[-1] - nav.index[0]).days
    years       = total_days / 365.25
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    cagr         = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1

    # ── Risk metrics ────────────────────────────────────────
    annual_vol = daily_returns.std() * np.sqrt(252)

    # Sharpe Ratio (risk-free rate ≈ 6.5% for India)
    rf_daily   = 0.065 / 252
    excess_ret = daily_returns - rf_daily
    sharpe     = (excess_ret.mean() / daily_returns.std()) * np.sqrt(252)

    # Sortino Ratio (downside deviation only)
    downside_ret = daily_returns[daily_returns < rf_daily]
    downside_std = downside_ret.std() * np.sqrt(252)
    sortino      = (cagr - 0.065) / (downside_std + 1e-9)

    # Max Drawdown
    rolling_max = nav.cummax()
    drawdown    = (nav - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate (% of positive months)
    monthly_returns = nav.resample("ME").last().pct_change().dropna()
    win_rate        = (monthly_returns > 0).mean()

    metrics = {
        "Total Return (%)":    round(total_return * 100, 2),
        "CAGR (%)":            round(cagr * 100, 2),
        "Annual Volatility (%)": round(annual_vol * 100, 2),
        "Sharpe Ratio":        round(sharpe, 3),
        "Sortino Ratio":       round(sortino, 3),
        "Max Drawdown (%)":    round(max_dd * 100, 2),
        "Calmar Ratio":        round(calmar, 3),
        "Monthly Win Rate (%)": round(win_rate * 100, 2),
    }

    # ── Alpha / Beta vs benchmark ───────────────────────────
    if benchmark_nav is not None:
        bench_nav    = benchmark_nav.reindex(nav.index, method="ffill").dropna()
        bench_ret    = bench_nav.pct_change().dropna()

        # Align
        common_idx   = daily_returns.index.intersection(bench_ret.index)
        port_ret_a   = daily_returns.loc[common_idx]
        bench_ret_a  = bench_ret.loc[common_idx]

        if len(common_idx) > 10:
            cov_matrix   = np.cov(port_ret_a, bench_ret_a)
            beta         = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-9)

            bench_cagr   = (bench_nav.iloc[-1] / bench_nav.iloc[0]) ** (1 / years) - 1
            alpha        = cagr - (0.065 + beta * (bench_cagr - 0.065))  # CAPM alpha

            correlation  = np.corrcoef(port_ret_a, bench_ret_a)[0, 1]
            tracking_err = (port_ret_a - bench_ret_a).std() * np.sqrt(252)
            info_ratio   = (cagr - bench_cagr) / (tracking_err + 1e-9)

            metrics.update({
                "Alpha (ann. %)":   round(alpha * 100, 2),
                "Beta":             round(beta, 3),
                "Correlation":      round(correlation, 3),
                "Tracking Error (%)": round(tracking_err * 100, 2),
                "Info Ratio":       round(info_ratio, 3),
                "Benchmark CAGR (%)": round(bench_cagr * 100, 2),
            })

    return metrics


def print_metrics(metrics: dict, title: str = "Strategy Performance"):
    """Pretty print metrics table."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

    groups = {
        "Returns":   ["Total Return (%)", "CAGR (%)", "Benchmark CAGR (%)"],
        "Risk":      ["Annual Volatility (%)", "Max Drawdown (%)", "Monthly Win Rate (%)"],
        "Risk-Adj":  ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
        "vs Bench":  ["Alpha (ann. %)", "Beta", "Correlation", "Info Ratio", "Tracking Error (%)"],
    }

    for group, keys in groups.items():
        printed_header = False
        for key in keys:
            if key in metrics:
                if not printed_header:
                    print(f"\n  {group}:")
                    printed_header = True
                print(f"    {key:<25}: {metrics[key]:>10}")

    print(f"\n{'='*50}\n")


def monthly_returns_table(nav: pd.Series) -> pd.DataFrame:
    """
    Build a Year × Month table of monthly returns (%).
    """
    monthly = nav.resample("ME").last().pct_change().dropna() * 100
    monthly.index = pd.to_datetime(monthly.index)

    table = pd.DataFrame({
        "year":  monthly.index.year,
        "month": monthly.index.strftime("%b"),
        "ret":   monthly.values,
    })

    pivot = table.pivot(index="year", columns="month", values="ret")

    # Order months correctly
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    # Add annual return column
    pivot["Annual"] = pivot.sum(axis=1, skipna=True)

    return pivot.round(2)
