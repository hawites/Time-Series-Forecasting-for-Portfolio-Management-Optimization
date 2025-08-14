# 📈 GMF — Time Series & Portfolio (Tasks 1–2)

Concise, OOP-based workflow for **data preprocessing & EDA** (Task 1) and **forecasting** (Task 2) on **TSLA**, **BND**, **SPY**. Outputs feed later **portfolio optimization** and **backtesting**.

---

## 🔧 Quick Setup
- Python **3.11** recommended (for TensorFlow/LSTM). ARIMA works on **3.13**.
- Install & test:
  ```bash
  pip install -r requirements.txt
  pytest -q
  ```

**Key folders**
```
src/ (config, data_loader, features, eda, splits, models/, utils/)
data/{raw,processed}
reports/{figures,interim}
notebooks/{Data_eda.ipynb, Modeling.ipynb}
```

---

## 📊 Task 1 — EDA & Preprocessing (Summary)

**Goal**
- Fetch prices via `yfinance` (2015-07-01 → 2025-07-31).
- Clean/align data, compute **daily simple** & **log returns**.
- Run focused EDA (trends, volatility, stationarity, basic risk).

**What we did**
- `DataLoader` saves per-ticker CSVs → `data/raw/`.
- `FeatureEngineer` pivots **Adj Close** to wide (`TSLA`, `BND`, `SPY`), coerces numerics, ffill/bfill gaps, adds `*_ret`, `*_logret`, exports `merged_features.csv`.
- `EDAAnalyzer`:
  - Price & returns plots.
  - Rolling mean/std (21/63/252).
  - ADF tests (prices non-stationary; returns stationary).
  - VaR (historical) & Sharpe (annualized).

**Outputs**
- Data: `data/processed/merged_features.csv`
- Figures: `reports/figures/closing_prices.png`, `daily_returns.png`, `tsla_rolling_stats.png`
- CSVs: `reports/interim/basic_stats.csv`, `reports/interim/risk_metrics.csv`

---

## 🤖 Task 2 — Modeling (ARIMA vs LSTM) (Summary)

**Goal**
- Forecast **TSLA returns**; compare **ARIMA** (classical) vs **LSTM** (DL, optional).
- Chronological split: **train ≤ 2023-12-31**, **test ≥ 2024-01-01**.
- Evaluate on **returns** and **reconstructed prices**.

**What we did**
- `TimeSeriesSplitter` for date-based splits.
- **ARIMAModel**:
  - AIC grid over `(p,d,q)`, `trend="n"` for returns.
  - Fits on RangeIndex to avoid freq warnings; predictions re-indexed to real dates.
- **LSTMModel** (optional if TF available):
  - Windowed univariate LSTM (`lookback=60`), EarlyStopping.
  - Skips gracefully if TensorFlow is not installed.

**Evaluation & Artifacts**
- Metrics (MAE/RMSE/MAPE) on returns & reconstructed prices → `reports/interim/task2_metrics.csv`
- Forecast plots:
  - `reports/figures/tsla_pred_price_arima.png`
  - `reports/figures/tsla_pred_price_lstm.png` *(if TF available)*

---

## ▶️ How to Run
1) **Task 1**: `notebooks/01_data_eda.ipynb` → creates processed data + EDA artifacts.
2) **Task 2**: `notebooks/02_modeling.ipynb` → trains ARIMA/LSTM, saves metrics & plots.
3) **Tests**: `pytest -q` (LSTM test auto-skips if TF is missing).

---

## 🧱 Notes & Fixes
- Use **NumPy-only** metrics (no sklearn) to avoid Windows/Python 3.13 DLL issues.
- For ARIMA, fit on **RangeIndex** then reattach dates to suppress frequency warnings.
- If `yfinance` not found in the notebook kernel, install with:
  ```python
  import sys; !{sys.executable} -m pip install yfinance
  ```

---

## ✅ Next Steps
- **Task 3**: 6–12 month forecast with confidence intervals + interpretation.
- **Task 4**: Efficient Frontier (PyPortfolioOpt); mark Tangency & Min-Vol; recommend weights.
- **Task 5**: Backtest vs **60/40 SPY/BND** (Aug 2024–Jul 2025); compare returns & Sharpe.
