# 📈 GMF — Time Series & Portfolio (Tasks 1–3)

OOP workflow for **data preprocessing & EDA** (Task 1), **modeling** (Task 2), and **6–12 month forecasting** (Task 3) on **TSLA**, **BND**, **SPY**. Outputs will feed **efficient frontier** optimization and **backtesting** next.

---

## 🔧 Quick Setup
- Python **3.11** recommended (TensorFlow/LSTM). ARIMA runs fine on **3.13**.
- Install & test:
  ```bash
  pip install -r requirements.txt
  pytest -q
  ```

**Key folders**
```
src/ (config, data_loader, features, eda, splits, forecast, models/, utils/)
data/{raw,processed}
reports/{figures,interim}
notebooks/{01_data_eda.ipynb, 02_modeling.ipynb, 03_forecast.ipynb}
```

---

## 📊 Task 1 — EDA & Preprocessing (Summary)
- `DataLoader` fetches yfinance data (**2015-07-01 → 2025-07-31**) to `data/raw/`.
- `FeatureEngineer` pivots **Adj Close** to wide (`TSLA`, `BND`, `SPY`), coerces numerics, ffill/bfill gaps; computes `*_ret`, `*_logret`; exports `data/processed/merged_features.csv`.
- `EDAAnalyzer` produces price/returns charts, rolling stats, **ADF** stationarity checks, **VaR** and **Sharpe**.
  - Prices: **non-stationary**; Returns: **stationary** → model **returns**.

**Artifacts**
- Data: `data/processed/merged_features.csv`
- Figures: `reports/figures/closing_prices.png`, `daily_returns.png`, `tsla_rolling_stats.png`
- CSVs: `reports/interim/basic_stats.csv`, `reports/interim/risk_metrics.csv`

---

## 🤖 Task 2 — Modeling (ARIMA vs LSTM) (Summary)
- Chronological split: **train ≤ 2023-12-31**, **test ≥ 2024-01-01**.
- **ARIMAModel**: AIC grid over `(p,d,q)` on **returns** (`trend="n"`); fit on **RangeIndex**, then re-attach dates (avoids freq warnings).
- **LSTMModel** (optional): univariate LSTM windowing (`lookback=60`); skipped if TensorFlow not installed.
- Evaluation on **returns** and **reconstructed prices** (MAE/RMSE/MAPE).  
  - In our run, ARIMA selected **(2,0,2)**.

**Artifacts**
- Figures: `reports/figures/tsla_pred_price_arima.png` (+ LSTM if available)
- CSV: `reports/interim/task2_metrics.csv`

---

## 🔮 Task 3 — 6–12 Month Forecasts (ARIMA on TSLA Returns)
We forecast **TSLA log returns**, then reconstruct **price paths** and **95% CI bands**.

### Results (from `reports/interim/forecast_summary.csv`)
| horizon | ARIMA order | mean_ret_annualized | ret_CI_wid_avg | px_end_mean | px_end_lo | px_end_hi |
|:------:|:-----------:|--------------------:|---------------:|------------:|----------:|----------:|
| 6m | (2, 0, 2) | -0.002249 | 0.141764 | 248.198215 | 0.023533 | 1,387,656.000000 |
| 12m | (2, 0, 2) | -0.001124 | 0.141779 | 248.198214 | 0.000002 | 7,771,179,000.000000 |

**Interpretation**
- **Mean drift**: ~flat/slightly negative (−0.11% to −0.23% annualized).
- **Return CI** averages ≈ **0.142** per day. Compounded into prices, this yields **very wide price CIs** (upper tails blow up), which is expected when compounding daily bounds directly.

**Best practice (applied in the memo going forward)**
- Emphasize **CI on returns**; for prices, prefer:
  1) Aggregate **log-returns** (sum) then exponentiate (lognormal assumption), or
  2) **Monte Carlo** fan charts using ARIMA residuals (5th–95th percentiles).

**Artifacts**
- Figures: `reports/figures/tsla_returns_forecast_6m.png`, `tsla_price_forecast_6m.png`, `tsla_returns_forecast_12m.png`, `tsla_price_forecast_12m.png`
- CSVs: `reports/interim/tsla_forecast_6m.csv`, `reports/interim/tsla_forecast_12m.csv`, `reports/interim/forecast_summary.csv`

---

## ▶️ How to Run
1) **Task 1** — `notebooks/Data_eda.ipynb` → processed data + EDA outputs.  
2) **Task 2** — `notebooks/Modeling.ipynb` → ARIMA/LSTM comparison; metrics/plots.  
3) **Task 3** — `notebooks/Forecast.ipynb` → 6m/12m forecasts with CI; summary CSV.

---

## ✅ Next Steps
- **Task 4 (MPT)**: Use TSLA **expected return** from Task 2/3; BND & SPY **historical annualized**; compute covariance; **Efficient Frontier**; mark **Tangency** & **Min-Vol**; recommend weights.
- **Task 5 (Backtest)**: Compare strategy vs **60/40 SPY/BND** over Aug‑2024 → Jul‑2025; report cumulative return & Sharpe.
