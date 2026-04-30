# Nifty ML Strategy

Machine-learning driven stock selection and backtesting framework for Indian equities, built around XGBoost and strict time-series validation.

## Why This Repo

This project demonstrates an end-to-end quant research workflow:

- data ingestion and caching for a Nifty universe
- leakage-safe feature engineering
- probabilistic stock ranking with XGBoost
- periodic portfolio construction and cost-aware backtesting
- institutional-style performance diagnostics and visual reporting

It is designed as both a research notebook replacement and a production-style script pipeline.

## Highlights

- **Model**: XGBoost binary classifier with `TimeSeriesSplit`
- **Signals**: momentum, mean reversion, volatility, volume, relative strength
- **Execution**: fixed schedule rebalancing with transaction cost modeling
- **Evaluation**: out-of-sample run with benchmark comparison
- **Deliverables**: NAV, trades, metrics, charts, feature importance

## Project Layout

```text
nifty_ml_strategy/
├── main.py          # Pipeline orchestration (download -> train -> backtest -> report)
├── config.py        # Universe, dates, rebalance, costs, and global settings
├── data_loader.py   # Historical data retrieval + caching
├── features.py      # Signal generation and supervised dataset construction
├── model.py         # Train/evaluate model and produce stock scores
├── backtest.py      # Rebalancing logic, portfolio simulation, benchmark NAV
├── metrics.py       # CAGR, Sharpe, drawdown, alpha and other KPIs
├── visualize.py     # Reporting charts and summary visuals
├── requirements.txt
├── data/            # Auto-created local data cache
└── output/          # Auto-generated outputs (csv, model, plots)
```

## Quickstart

### 1) Clone and enter

```bash
git clone <your-repo-url>
cd nifty_ml_strategy
```

### 2) Create environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3) macOS prerequisite (for XGBoost)

```bash
brew install libomp
```

### 4) Run full pipeline

```bash
python main.py
```

## CLI Options

```bash
# full run
python main.py

# use cached market data
python main.py --skip-dl

# use cached data + pre-trained model
python main.py --skip-dl --skip-train

# quicker model search
python main.py --n-iter 10
```

## Pipeline at a Glance

1. Download market data and benchmark series  
2. Build model features and target labels  
3. Train XGBoost with time-aware cross-validation  
4. Score each rebalance date and select top-ranked stocks  
5. Simulate portfolio with turnover costs  
6. Compute metrics and generate visual artifacts  

## Outputs

After execution, `output/` includes:

- `nav.csv` - Daily strategy NAV
- `trades.csv` - Rebalance-level trade log
- `metrics.txt` - Performance summary
- `model.pkl` - Trained model pipeline
- `nav_chart.png` - Strategy vs benchmark curve
- `metrics_table.png` - KPI summary table
- `monthly_heatmap.png` - Monthly return heatmap
- `feature_importance.png` - Signal importance ranking
- `rolling_sharpe.png` - Rolling Sharpe profile

## Tunable Parameters

Update `config.py` to customize:

- universe constituents
- train/test split window
- rebalance frequency
- number of selected stocks
- transaction cost assumptions
- forecast horizon for target labels

## Known Limitations

- **Survivorship bias**: historical index membership changes are simplified
- **Data quality constraints**: public data feeds can have gaps/outliers
- **Long-only implementation**: no short book or market-neutral overlay
- **Simplified slippage model**: flat costs, no liquidity curve simulation

## Roadmap Ideas

- walk-forward retraining schedule
- probability calibration and confidence thresholds
- sector/industry neutrality constraints
- factor exposure controls and risk budgeting
- richer transaction cost model with turnover caps

## Disclaimer

This project is for educational and research purposes only, not financial advice.
