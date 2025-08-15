# 📈 GMF — Time Series & Portfolio (Tasks 1–4)

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

# 🚀 Task 4: Portfolio Optimization (Modern Portfolio Theory)

This task builds a **three-asset portfolio** (**TSLA**, **BND**, **SPY**) using **Modern Portfolio Theory (MPT)**.  
You will construct expected returns and covariance, compute the **Efficient Frontier**, and identify the **Maximum Sharpe (Tangency)** and **Minimum Volatility** portfolios.  
Artifacts include a frontier plot, portfolio weights, and performance stats for decision-making and later **backtesting** (Task 5).

---

## 🎯 Objective
- Translate model signals (Task 2–3) and historical data into **portfolio weights**.
- Generate the **Efficient Frontier** and mark **Max Sharpe** & **Min Vol** portfolios.
- Export weights and performance metrics for reporting and backtesting.

---

## 📦 Inputs
- **Processed features** (from Task 1):  
  `data/processed/merged_features.csv` (contains daily simple returns: `TSLA_ret`, `BND_ret`, `SPY_ret`)
- **TSLA 12-month forecast (optional)** (from Task 3):  
  `reports/interim/tsla_forecast_12m.csv` (uses `ret_mean` column of daily returns)  
  > If missing, the optimizer **falls back** to TSLA’s **historical annualized mean**.
- **Risk-free rate**: from `src/config.py → Settings.risk_free_rate` (annualized, e.g., 0.045)

---

## 🧠 Methodology

### 1) Expected Returns (annualized vector **μ**)
- **TSLA**: If `tsla_forecast_12m.csv` exists, compute `mean(ret_mean) × 252`. Otherwise, historical daily mean × 252.  
- **BND & SPY**: Historical daily mean × 252.

### 2) Covariance (annualized matrix **Σ**)
- Use historical daily returns of **TSLA**, **BND**, **SPY** → sample covariance × 252.  
  > Optional enhancement (not required here): consider shrinkage (Ledoit–Wolf) via `pypfopt.risk_models` for more stable Σ.

### 3) Optimization (PyPortfolioOpt)
- **Efficient Frontier** traced by sweeping target returns.  
- **Max Sharpe**: Tangency portfolio maximizing `(μ_p − r_f)/σ_p`.  
- **Min Volatility**: Portfolio minimizing σ_p independent of μ.

### 4) Outputs
- Frontier points (**μ**, **σ**), **Max Sharpe** & **Min Vol** metrics.
- Cleaned weights (sum to 1, small values zeroed).

---

## 🗂 Files & Modules

**Code**
- `src/portfolio/optimizer.py` → `PortfolioOptimizer` (build μ & Σ, frontier, max-sharpe, min-vol)
- `src/utils/plotting.py` → `Plotter.efficient_frontier(...)` (frontier + markers)

**Notebook**
- `notebooks/Portfolio.ipynb` (run end-to-end; saves all artifacts)

**Tests**
- `tests/test_portfolio_optimizer.py` (synthetic sanity checks)

---

## ▶️ How to Run

1. **Install dependencies**
   ```bash
   pip install PyPortfolioOpt
   ```

2. **Open** `notebooks/Portfolio.ipynb` and run all cells:
   - Loads `merged_features.csv`
   - Builds expected returns (μ) and covariance (Σ)
   - Computes **Efficient Frontier**, **Max Sharpe**, **Min Vol**
   - Saves weights/metrics and frontier figure

3. **Artifacts written to**
   - **Figure**: `reports/figures/efficient_frontier.png`
   - **Weights**:  
     - `reports/interim/weights_max_sharpe.csv`  
     - `reports/interim/weights_min_vol.csv`
   - **Stats**: `reports/interim/portfolio_stats.csv` (μ, σ, Sharpe for both portfolios)

---

## 📊 Interpreting the Results

- **Max Sharpe (Tangency)**: Highest expected risk‑adjusted return.  
  Typical outcome: overweight **SPY**, tuned **TSLA**, and **BND** as ballast, depending on inputs.
- **Min Volatility**: Stability‑first.  
  Typical outcome: larger **BND** weight, modest **SPY**, minimal **TSLA**.

> **Note on your Task 3 result:** If the 12‑month TSLA expected return is **near zero or negative**, the optimizer will **naturally reduce TSLA weight** in Max Sharpe and Min Vol portfolios. This is expected behavior and reflects a **risk‑aware** stance.

---

## 🧪 Sanity Checks

- **Weights sum to 1** (check CSV).  
- **Sharpe** should be **higher** for the Max Sharpe portfolio than Min Vol.  
- **Frontier** must slope upward (monotonic μ with σ), aside from minor numerical noise.  
- If numerical issues occur (e.g., singular Σ), consider:  
  - Increasing lookback history, or  
  - Using shrinkage covariance (`risk_models.CovarianceShrinkage`).

---

## ⚙️ Git Workflow (suggested)

```bash
git checkout -b feat/task4-portfolio-optimization

git add src/portfolio/optimizer.py tests/test_portfolio_optimizer.py
git commit -m "feat(task4): MPT optimizer + tests"

git add src/utils/plotting.py
git commit -m "feat(task4): efficient frontier plotting with markers"

git add notebooks/04_portfolio.ipynb
git commit -m "docs(task4): portfolio optimization notebook (frontier, max-sharpe, min-vol)"
```

---

## 📌 Reporting Snippets (paste into memo)

**Method:** μ from forecast+history, Σ from historical, MPT via PyPortfolioOpt.  
**Key Portfolios:** Max Sharpe (tangency) and Min Vol.  
**Recommendation:** Choose **[Max Sharpe / Min Vol]** based on mandate and risk tolerance; if TSLA forecast weak/negative, tilt **toward SPY/BND** to preserve Sharpe.

---

## 🔭 Next (Task 5: Backtesting)

- Use chosen weights to **simulate performance** vs a **60/40 SPY/BND** benchmark over **Aug‑2024 → Jul‑2025**.  
- Plot cumulative returns; compute annualized return, volatility, and Sharpe; summarize relative performance.





