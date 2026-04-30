# ============================================================
# visualize.py — All charts for the strategy report
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from config import OUTPUT_DIR

STYLE = {
    "strategy":  "#1a5f7a",
    "benchmark": "#e07b39",
    "positive":  "#2ecc71",
    "negative":  "#e74c3c",
    "grid":      "#e8ecef",
    "bg":        "#fafafa",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _pct_fmt(x, pos):
    return f"{x:.0f}%"


def plot_nav(nav: pd.Series,
             benchmark_nav: pd.Series,
             metrics: dict,
             save: bool = True):
    """
    Main NAV chart with drawdown panel.
    """
    fig = plt.figure(figsize=(14, 9), facecolor=STYLE["bg"])
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.2, 1], hspace=0.08)

    # ── Panel 1: NAV ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(nav.index, nav.values, color=STYLE["strategy"],  lw=2,   label="ML Strategy")
    ax1.plot(benchmark_nav.reindex(nav.index, method="ffill").index,
             benchmark_nav.reindex(nav.index, method="ffill").values,
             color=STYLE["benchmark"], lw=1.5, linestyle="--", label="Nifty 50", alpha=0.9)

    ax1.set_facecolor(STYLE["bg"])
    ax1.set_ylabel("NAV (Base = 100)", fontsize=11)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax1.grid(True, color=STYLE["grid"], linewidth=0.7)
    ax1.set_title("Nifty ML Strategy — Out-of-Sample Performance (2019–2024)",
                  fontsize=13, fontweight="bold", pad=12)

    # Annotation: final values
    ax1.annotate(f"Strategy: {nav.iloc[-1]:.0f}",
                 xy=(nav.index[-1], nav.iloc[-1]),
                 xytext=(-100, 10), textcoords="offset points",
                 fontsize=9, color=STYLE["strategy"],
                 arrowprops=dict(arrowstyle="-", color=STYLE["strategy"], lw=0.8))

    # ── Panel 2: Drawdown ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    rolling_max = nav.cummax()
    drawdown    = (nav - rolling_max) / rolling_max * 100

    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color=STYLE["negative"], alpha=0.5, label="Drawdown")
    ax2.plot(drawdown.index, drawdown.values, color=STYLE["negative"], lw=0.8)
    ax2.set_facecolor(STYLE["bg"])
    ax2.set_ylabel("Drawdown %", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax2.grid(True, color=STYLE["grid"], linewidth=0.7)
    ax2.axhline(0, color="black", lw=0.5)

    # ── Panel 3: Monthly Returns Bar ────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    monthly = nav.resample("ME").last().pct_change().dropna() * 100
    colors  = [STYLE["positive"] if r >= 0 else STYLE["negative"] for r in monthly]
    ax3.bar(monthly.index, monthly.values, width=20, color=colors, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_facecolor(STYLE["bg"])
    ax3.set_ylabel("Monthly %", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax3.grid(True, color=STYLE["grid"], linewidth=0.7, axis="y")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    if save:
        path = os.path.join(OUTPUT_DIR, "nav_chart.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    plt.close()


def plot_metrics_table(metrics: dict, save: bool = True):
    """
    Render metrics as a clean table image.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=STYLE["bg"])
    ax.axis("off")

    rows = [[k, str(v)] for k, v in metrics.items()]
    headers = ["Metric", "Value"]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 1.8)

    # Style header
    for col in range(2):
        table[0, col].set_facecolor(STYLE["strategy"])
        table[0, col].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for row in range(1, len(rows) + 1):
        color = "#eaf2f8" if row % 2 == 0 else "white"
        for col in range(2):
            table[row, col].set_facecolor(color)

    ax.set_title("Strategy Performance Summary", fontsize=13,
                 fontweight="bold", pad=20, color=STYLE["strategy"])

    if save:
        path = os.path.join(OUTPUT_DIR, "metrics_table.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    plt.close()


def plot_monthly_heatmap(nav: pd.Series, save: bool = True):
    """
    Year × Month heatmap of monthly returns.
    """
    from metrics import monthly_returns_table

    table = monthly_returns_table(nav)

    fig, ax = plt.subplots(figsize=(14, max(4, len(table) * 0.7)),
                           facecolor=STYLE["bg"])

    vmax = max(abs(table.values[np.isfinite(table.values)].max()), 5)

    im = ax.imshow(table.values.astype(float), cmap="RdYlGn",
                   aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, fontsize=10)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=10)

    # Annotate cells
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            val = table.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Return %")
    ax.set_title("Monthly Returns Heatmap", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    if save:
        path = os.path.join(OUTPUT_DIR, "monthly_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    plt.close()


def plot_feature_importance(pipeline, save: bool = True):
    """Bar chart of XGBoost feature importances."""
    from features import FEATURE_COLS

    model       = pipeline.named_steps["model"]
    importances = model.feature_importances_

    fi = pd.Series(importances, index=FEATURE_COLS).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=STYLE["bg"])
    bars = ax.barh(fi.index, fi.values, color=STYLE["strategy"], alpha=0.8)
    ax.set_facecolor(STYLE["bg"])
    ax.set_xlabel("Feature Importance (XGBoost)", fontsize=11)
    ax.set_title("Signal Importance", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="x", color=STYLE["grid"], linewidth=0.7)

    # Annotate values
    for bar, val in zip(bars, fi.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    if save:
        path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    plt.close()


def plot_rolling_sharpe(nav: pd.Series, window: int = 126, save: bool = True):
    """Rolling 6-month Sharpe Ratio."""
    daily_ret = nav.pct_change().dropna()
    rf_daily  = 0.065 / 252

    rolling_sharpe = (
        (daily_ret - rf_daily)
        .rolling(window)
        .mean()
        .div(daily_ret.rolling(window).std())
        * np.sqrt(252)
    ).dropna()

    fig, ax = plt.subplots(figsize=(13, 4), facecolor=STYLE["bg"])
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=STYLE["strategy"], lw=1.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.axhline(1, color=STYLE["positive"], lw=0.8, linestyle="--", alpha=0.7, label="Sharpe = 1")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values >= 0,
                    color=STYLE["positive"], alpha=0.2)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values < 0,
                    color=STYLE["negative"], alpha=0.2)
    ax.set_facecolor(STYLE["bg"])
    ax.set_ylabel("Rolling Sharpe (6M)", fontsize=11)
    ax.set_title("Rolling 6-Month Sharpe Ratio", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, color=STYLE["grid"], linewidth=0.7)

    if save:
        path = os.path.join(OUTPUT_DIR, "rolling_sharpe.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")

    plt.close()
