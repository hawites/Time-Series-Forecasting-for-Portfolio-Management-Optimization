from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2024-08-01"
    end: str = "2025-07-31"
    rebalance: Literal["none", "monthly"] = "none"  # hold or monthly reset to target weights
    rf_annual: float = 0.045  # annual risk-free for Sharpe

@dataclass(frozen=True)
class BacktestResult:
    cumrets: pd.DataFrame   # columns: ["strategy","benchmark"]
    daily: pd.DataFrame     # columns: ["strategy","benchmark"]
    stats: pd.DataFrame     # index: ["strategy","benchmark"], cols: ["annual_ret","annual_vol","sharpe"]

class Backtester:
   

    def __init__(self, returns_df: pd.DataFrame, cfg: Optional[BacktestConfig] = None) -> None:
       
        self.returns = returns_df.astype(float).sort_index()
        self.cfg = cfg or BacktestConfig()

    @staticmethod
    def _to_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[pd.to_datetime(start): pd.to_datetime(end)]

    def _simulate_path(self, ret: pd.DataFrame, weights: Dict[str,float], rebalance: str) -> pd.Series:
       
        tickers = list(weights.keys())
        R = ret[tickers].copy()
        w = np.array([weights[t] for t in tickers], dtype=float)
        w = w / w.sum()

        # start with $1 split by target weights
        alloc = w.copy()  # dollar alloc since PV=1 initially
        pv_path = []
        last_month = R.index[0].month

        for dt, row in R.iterrows():
            # update each sleeve by (1 + r_it)
            alloc = alloc * (1.0 + row.values)
            pv = float(alloc.sum())
            pv_path.append(pv)

            # rebalance at month change if configured
            if rebalance == "monthly" and dt.month != last_month:
                alloc = pv * w  # reset sleeves to target weights
            last_month = dt.month

        pv_series = pd.Series(pv_path, index=R.index, name="pv")
        # convert to daily simple returns from PV path
        daily = pv_series.pct_change().fillna(0.0)
        return daily

    @staticmethod
    def _annualize(daily_ret: pd.Series, rf_annual: float) -> Dict[str, float]:
        mu_d = float(daily_ret.mean())
        sd_d = float(daily_ret.std())
        ann_ret = (1.0 + mu_d) ** 252 - 1.0
        ann_vol = sd_d * (252.0 ** 0.5)
        sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan
        return {"annual_ret": ann_ret, "annual_vol": ann_vol, "sharpe": sharpe}

    def run(
        self,
        strategy_weights: Dict[str,float],
        benchmark_weights: Optional[Dict[str,float]] = None
    ) -> BacktestResult:

        # Slice the backtest window
        R = self._to_period(self.returns, self.cfg.start, self.cfg.end)

        # Strategy daily returns
        strat_daily = self._simulate_path(R, strategy_weights, self.cfg.rebalance)

        # Benchmark daily returns (default 60/40 SPY/BND)
        if benchmark_weights is None:
            benchmark_weights = {"SPY": 0.60, "BND": 0.40}
        # ensure all required columns exist
        missing = [t for t in benchmark_weights.keys() if t not in R.columns]
        if missing:
            raise ValueError(f"Benchmark missing assets in returns_df: {missing}")

        bench_daily = self._simulate_path(R, benchmark_weights, self.cfg.rebalance)

        # Assemble frames
        daily = pd.concat([strat_daily.rename("strategy"), bench_daily.rename("benchmark")], axis=1)
        cum = (1.0 + daily).cumprod()

        # Stats
        s_stats = self._annualize(daily["strategy"], self.cfg.rf_annual)
        b_stats = self._annualize(daily["benchmark"], self.cfg.rf_annual)
        stats = pd.DataFrame([s_stats, b_stats], index=["strategy","benchmark"])

        return BacktestResult(cumrets=cum, daily=daily, stats=stats)
