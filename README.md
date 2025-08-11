# 📊 Task 1: Exploratory Data Analysis and Preprocessing

## 🔍 Objective
The purpose of this task was to:
- Fetch **historical market data** for **TSLA**, **BND**, and **SPY** using `yfinance`.
- Preprocess and align the datasets; derive **daily simple** and **log returns**.
- Perform **EDA** (trends, volatility, stationarity) and compute basic **risk metrics**.
- Produce a clean, analysis-ready dataset for **Task 2 modeling** (ARIMA/LSTM) and later **portfolio optimization**.

---

## 🗂 Data Loading
- Implemented via **OOP** `DataLoader` class.
- Download range: **2015-07-01 → 2025-07-31** (observed date range in data: **2015-07-02 → 2025-07-30**).
- Saved per-ticker CSVs to `data/raw/<TICKER>.csv`, then merged later in processing.
- Resulting aligned dataset contains **~2,534 trading days** per asset (after cleaning/alignment).

**Key files/classes**
- `src/config.py` → `Settings` (dates, tickers, paths, risk-free rate)
- `src/data_loader.py` → `DataLoader.fetch_and_cache()` / `load_all_list()`

---

## 🧼 Data Cleaning Strategy
Performed in **`FeatureEngineer`**:

### ✔️ Column & Index Standardization
- Parsed `Date` as `datetime` and used it as the index; sorted chronologically.

### ✔️ Asset Alignment & Missing Day Handling
- Pivoted to wide format on **Adj Close** → columns: `TSLA`, `BND`, `SPY`.
- Applied **forward-fill** then **back-fill** to handle asset-specific holidays and occasional gaps.

### ✔️ Return Calculations
- **Simple returns**: `pct_change()` → `*_ret`.
- **Log returns**: `log(AdjClose_t / AdjClose_{t-1})` → `*_logret`.

### ✔️ Numeric Safety
- Coerced price/volume columns to numeric (`errors='coerce'`), replaced ±∞ with `NaN`, and filled remaining gaps before computing returns.

> ℹ️ **Scaling/normalization** is deferred to Task 2 and will be applied **after** the train/test split to avoid leakage.

**Key files/classes**
- `src/features.py` → `FeatureEngineer.pipeline()` saves `data/processed/merged_features.csv`

---

## 📊 Exploratory Data Analysis (EDA)

### Prices & Returns
- Generated line plots for **Adj Close** (`TSLA`, `BND`, `SPY`) and **Daily Returns**.
- Computed **rolling statistics** (TSLA returns) over **21/63/252** trading days to visualize short- and long-term volatility.

### Stationarity (ADF)
- **TSLA price**: `ADF ≈ -1.4195`, **p-value ≈ 0.5729** → *non-stationary* (as expected for price levels).
- **TSLA log returns**: `ADF ≈ -11.7171`, **p-value ≈ 1.44e-21** → *stationary*, suitable for ARIMA/LSTM modeling.

### Descriptive Stats (sample)
- Returns summary (count = **2,534** for each asset):
  - `TSLA_ret` mean **0.001828**, std **0.037285**, min **-0.2106**, max **0.2269**
  - `BND_ret` mean **0.000078**, std **0.003460**
  - `SPY_ret` mean **0.000575**, std **0.011491**

> Only plots with valid and sufficient data were rendered.

**Key files/classes**
- `src/eda.py` → `EDAAnalyzer` (ADF, rolling stats, Sharpe, VaR)  
- `src/utils/plotting.py` → `Plotter` (saves to `reports/figures/`)

---

## 📈 Risk Metrics Summary (Daily → Annualized where relevant)

| Ticker | VaR_95 (daily) | Sharpe (annual) | Est. Annual Return | Est. Annual Vol |
|:-----:|:---------------:|:----------------:|:------------------:|:----------------:|
| TSLA  | **-0.054663**   | **0.744550**     | **0.460690**       | **0.591887**     |
| BND   | **-0.004900**   | **-0.007296**    | **0.019599**       | **0.054919**     |
| SPY   | **-0.017195**   | **0.684427**     | **0.144844**       | **0.182407**     |

**Interpretation (high level):**
- **TSLA**: highest expected return with high volatility; positive Sharpe.
- **SPY**: solid risk-adjusted profile; positive Sharpe.
- **BND**: low volatility & low return; near-zero Sharpe in this window (behaves as stabilizer/ballast).

---

## 📁 Output Artifacts

**Data**
- Cleaned features (prices + returns):  
  `data/processed/merged_features.csv`

**Figures**
- `reports/figures/closing_prices.png`
- `reports/figures/daily_returns.png`
- `reports/figures/tsla_rolling_stats.png`

**Tables / CSV Snapshots**
- `reports/interim/basic_stats.csv`
- `reports/interim/risk_metrics.csv`

---

## ✅ Summary
Task 1 successfully:
- Downloaded and aligned multi-asset time series (TSLA, BND, SPY) with robust numeric handling.
- Created an analysis-ready dataset with **Adj Close**, **simple returns**, and **log returns**.
- Validated modeling assumptions: **prices non-stationary**, **returns stationary**.
- Produced informative **EDA figures** and **risk metrics** to guide modeling and portfolio optimization.

**Next:** Proceed to **Task 2 — Time Series Modeling (ARIMA/SARIMA vs LSTM)** using a **chronological split** and evaluating **MAE/RMSE/MAPE** to select the best forecaster for 6–12 month projections.
