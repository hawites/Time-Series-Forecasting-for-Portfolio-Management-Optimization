# 📚 GMF Investments — Time Series & Portfolio Optimization (Tasks 1–5)

End-to-end, **OOP-structured** project that fetches market data, engineers features, builds/compares forecasters, produces **6–12 month** projections, optimizes a **TSLA–BND–SPY** portfolio (MPT), and **backtests** vs a 60/40 benchmark.

> Stack: `yfinance`, `pandas`, `statsmodels (ARIMA)`, optional `tensorflow` (LSTM), `PyPortfolioOpt` (MPT).  
> Notebooks call into modules under `src/` and save reproducible artifacts.

---

## 🔧 Quick Setup

- Python **3.11** recommended (for TensorFlow/LSTM). ARIMA & Tasks 1/4/5 also work on **3.13**.
- Install & test:
  ```bash
  pip install -r requirements.txt
  pytest -q
  ```
- Optional: install TensorFlow if you will run the LSTM model.

**Key structure**
```
src/
  config.py
  data_loader.py
  features.py
  eda.py
  splits.py
  forecast.py
  models/
    arima_model.py
    lstm_model.py
  portfolio/optimizer.py
  backtest/backtester.py
  utils/
    metrics.py
    plotting.py
data/
  raw/            # per-ticker yfinance cache
  processed/      # merged_features.csv
reports/
  figures/        # PNGs for EDA, forecasts, frontier, backtest
  interim/        # CSV snapshots (stats, metrics, forecasts, weights, backtest)
notebooks/
  Data_eda.ipynb
  Modeling.ipynb
  Forecast.ipynb
  Portfolio.ipynb
  Backtest.ipynb
tests/
```

---

# 📊 Task 1 — Exploratory Data Analysis & Preprocessing

### 🎯 Objective
- Fetch **TSLA**, **BND**, **SPY** daily data (Adj Close) via `yfinance` for **2015-07-01 → 2025-07-31**.
- Clean/align series; compute **daily simple** (`*_ret`) and **log returns** (`*_logret`).
- Run EDA (trends, volatility, stationarity) + basic risk metrics.

### 🗂 Implementation
- `DataLoader.fetch_and_cache()` saves per-ticker CSVs to `data/raw/`.
- `FeatureEngineer.pipeline()` pivots Adj Close to wide (`TSLA`, `BND`, `SPY`), coerces numerics, fixes ±∞, `ffill→bfill` gaps, derives returns, exports:
  - `data/processed/merged_features.csv`

### 🔍 EDA & Metrics
- Plots: `closing_prices.png`, `daily_returns.png`, `tsla_rolling_stats.png` (21/63/252-day windows).
- **ADF**: prices **non-stationary**, returns **stationary** (e.g., TSLA log-returns p≈1.4e-21).
- **Risk** saved to `reports/interim/risk_metrics.csv`: VaR₉₅ (daily), annual Sharpe, annualized return/vol.

**Artifacts**
- Data: `data/processed/merged_features.csv`  
- Figures: `reports/figures/closing_prices.png`, `daily_returns.png`, `tsla_rolling_stats.png`  
- Tables: `reports/interim/basic_stats.csv`, `reports/interim/risk_metrics.csv`

---

# 🤖 Task 2 — Modeling (ARIMA vs LSTM)

### 🎯 Objective
- Chronological split: **train ≤ 2023-12-31**, **test ≥ 2024-01-01**.
- Compare **ARIMA** (classical) vs **LSTM** (optional) on **TSLA returns** (prefer `TSLA_logret`).

### 🧩 Implementation
- `ARIMAModel`: AIC grid search over `(p,d,q)`; fit on **RangeIndex** to suppress date-freq warnings; reattach real dates for plots.
- `LSTMModel` (optional): univariate windowed LSTM (lookback=60). Skips gracefully if TensorFlow not installed.
- Metrics on **returns** and **reconstructed prices** (MAE/RMSE/MAPE). Price path: `P̂_t = P_train_last × ∏(1 + r̂_t)`.

**Artifacts**
- `reports/interim/task2_metrics.csv`  
- `reports/figures/tsla_pred_price_arima.png` (+ LSTM plot if available)

**Notes**
- For returns, `trend="n"` and **d=0** are typical.  
- Convergence warnings during ARIMA search are normal; the class skips non-convergent combos and retries with higher maxiter.

---

# 🔮 Task 3 — 6–12 Month Forecasts (ARIMA on Returns)

### 🎯 Objective
- Forecast **TSLA log returns** 6 & 12 months ahead; produce **95% CI** bands; reconstruct price paths.

### 🧩 Implementation
- `ARIMAForecaster` fits on returns (integer index), forecasts `steps` daily returns + CI, then reconstructs mean/CI price paths from the last train price.
- Notebook saves **CSV** and **plots** for both horizons.

### 📈 Results Summary (from your run)
`reports/interim/forecast_summary.csv`:

| horizon | ARIMA order | mean_ret_annualized | ret_CI_wid_avg | px_end_mean | px_end_lo | px_end_hi |
|:------:|:-----------:|--------------------:|---------------:|------------:|----------:|----------:|
| 6m | (2, 0, 2) | -0.002249 | 0.141764 | 248.198215 | 0.023533 | 1,387,656.000000 |
| 12m | (2, 0, 2) | -0.001124 | 0.141779 | 248.198214 | 0.000002 | 7,771,179,000.000000 |

**Interpretation**
- Mean drift ~flat/slightly negative (annualized).  
- **Return CI** ≈ 0.142/day; compounding daily **upper/lower** bounds yields **very wide price CIs** (upper tails explode). Expected with naive compounding.

**Better practice for prices**
- Prefer **CI on returns**, and for prices use either:
  - Sum of **log-returns** → exponentiate (lognormal assumption), or
  - **Monte Carlo** fan charts using ARIMA residuals (5–95th percentiles).

**Artifacts**
- Figures: `tsla_returns_forecast_6m.png`, `tsla_price_forecast_6m.png`, `tsla_returns_forecast_12m.png`, `tsla_price_forecast_12m.png`  
- CSVs: `tsla_forecast_6m.csv`, `tsla_forecast_12m.csv`, `forecast_summary.csv`

---

# 🧮 Task 4 — Portfolio Optimization (MPT)

### 🎯 Objective
Turn forecasts + history into **weights**; compute **Efficient Frontier**; mark **Max Sharpe** (tangency) and **Min Vol** portfolios.

### 🧩 Implementation
- Expected returns **μ (annualized)**:
  - **TSLA**: mean of Task-3 daily `ret_mean` × 252 (fallback: historical mean×252)
  - **BND & SPY**: historical daily mean × 252
- Covariance **Σ (annualized)**: sample cov of daily returns × 252
- Optimization using `PyPortfolioOpt`:
  - `max_sharpe(rf)` and `min_volatility()`
  - Frontier traced by sweeping target μ

**Artifacts**
- Figure: `reports/figures/efficient_frontier.png`  
- Weights: `reports/interim/weights_max_sharpe.csv`, `weights_min_vol.csv`  
- Stats: `reports/interim/portfolio_stats.csv` (μ, σ, Sharpe per portfolio)

**Recommendation pattern**
- Choose **Max Sharpe** unless σ is beyond mandate (e.g., > 1.5× SPY vol), then **Min Vol**. Document rationale in memo.

---

# 🧪 Task 5 — Backtesting (Strategy vs 60/40)

### 🎯 Objective
Simulate the selected portfolio vs a **60% SPY / 40% BND** benchmark over **Aug-2024 → Jul-2025**.

### 🧩 Implementation
- `Backtester` simulates **buy-and-hold** (`rebalance="none"`) or **monthly** rebalancing.
- Inputs: daily simple returns (`TSLA_ret`, `BND_ret`, `SPY_ret`) → renamed to `TSLA/BND/SPY`.
- Strategy weights from Task-4 CSV (Max Sharpe by default; fallback to Min Vol).

**Artifacts**
- Figure: `reports/figures/backtest_cumreturns.png`  
- CSVs: `reports/interim/backtest_daily_returns.csv`, `backtest_cumulative_curves.csv`, `backtest_stats.csv`

**Reported metrics**
- Annualized return, annualized volatility, and **Sharpe** (uses `Settings.risk_free_rate`).  
- Conclusion: “Outperformed” if strategy Sharpe > benchmark Sharpe (state both numbers).

---

## ▶️ How to Run (Notebooks)

1. **Task 1** — `notebooks/Data_eda.ipynb` → processed data + EDA artifacts  
2. **Task 2** — `notebooks/Modeling.ipynb` → ARIMA vs LSTM; metrics & plots  
3. **Task 3** — `notebooks/Forecast.ipynb` → 6m/12m forecasts; CI bands; summary CSV  
4. **Task 4** — `notebooks/Portfolio.ipynb` → Efficient Frontier; Max Sharpe & Min Vol; weights & stats  
5. **Task 5** — `notebooks/Backtest.ipynb` → backtest vs 60/40; cumulative curves & stats

---

## 🧱 Known Pitfalls & Fixes

- **ARIMA warnings & convergence**: Fit on **RangeIndex**; constrain grid for returns (`d=0`, small `p,q`); skip/retry on non-convergence.  
- **Date frequency warnings**: Avoid via integer index during fit; reattach dates for plotting.  
- **Type errors in returns**: Coerce numerics (`errors="coerce"`), replace ±∞, `ffill→bfill` before `pct_change()`.  
- **TensorFlow on Windows**: Use Python **3.11** env; LSTM optional and skipped if TF absent.  
- **Exploding price CIs**: Don’t compound daily min/max; use log-return aggregation or Monte Carlo.

---



## 📌 Memo Snippets (paste into Investment Memo)

- **Methodology:** μ from forecast (TSLA) + history (BND/SPY); Σ from historical; MPT for weights; backtest vs 60/40.  
- **Task 3:** CI widening over horizon; price CI caveat; focus on return bands.  
- **Task 4:** Present Max Sharpe & Min Vol weights and stats; recommend based on risk mandate.  
- **Task 5:** Summarize Sharpe/return/vol vs benchmark; state rebalancing policy & window.

---

**Reminder:** Models are **inputs** to a broader decision framework. Treat forecasts as **probabilistic**, stress-test assumptions, and triangulate with macro/valuation insight.
