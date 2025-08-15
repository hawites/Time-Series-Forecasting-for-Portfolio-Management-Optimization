from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

@dataclass
class PortfolioInputs:
    tickers: List[str]
    exp_returns_ann: pd.Series   # annualized expected returns (index=ticker)
    cov_ann: pd.DataFrame        # annualized covariance (tickers x tickers)
    rf_rate: float               # annual risk-free rate (e.g., 0.045)

class PortfolioOptimizer:
    

    def __init__(self, rf_rate: float = 0.045) -> None:
        self.rf_rate = float(rf_rate)

    @staticmethod
    def _annualize_mean_daily(ret: pd.Series) -> float:
        ret = pd.Series(ret).dropna().astype(float)
        return float(ret.mean() * 252)

    @staticmethod
    def _annualize_cov_daily(returns_df: pd.DataFrame) -> pd.DataFrame:
        r = returns_df.dropna(how="any").astype(float)
        return r.cov() * 252.0

    def build_expected_returns(
        self,
        returns_df: pd.DataFrame,
        use_tickers: List[str],
        tsla_forecast_csv: Optional[str] = None,
        tsla_mode: str = "forecast_12m",  # "forecast_12m" | "historical"
    ) -> pd.Series:
        
        rets = returns_df[ [f"{t}_ret" for t in use_tickers] ].copy()
        rets.columns = use_tickers

        exp = {}
        # TSLA from forecast if available
        if "TSLA" in use_tickers and tsla_forecast_csv and tsla_mode == "forecast_12m":
            try:
                f12 = pd.read_csv(tsla_forecast_csv, parse_dates=[0], index_col=0)
                # expect 'ret_mean' column of daily returns (log or simple). We used logret earlier,
                # but for expected return, simple daily mean is fine; if log, it's very close for small values.
                tsla_ann = float(pd.Series(f12["ret_mean"]).dropna().mean() * 252)
                exp["TSLA"] = tsla_ann
            except Exception:
                exp["TSLA"] = self._annualize_mean_daily(rets["TSLA"])
        # Others (and fallback)
        for t in use_tickers:
            if t == "TSLA" and t in exp:
                continue
            exp[t] = self._annualize_mean_daily(rets[t])

        return pd.Series(exp)[use_tickers]

    def build_covariance(
        self,
        returns_df: pd.DataFrame,
        use_tickers: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
       
        cols = [f"{t}_ret" for t in use_tickers]
        r = returns_df[cols].copy()
        r.columns = use_tickers
        if start:
            r = r.loc[pd.to_datetime(start):]
        if end:
            r = r.loc[:pd.to_datetime(end)]
        return self._annualize_cov_daily(r)

    def efficient_frontier(
        self, inputs: PortfolioInputs, n_points: int = 50
    ) -> pd.DataFrame:
        
        mu = inputs.exp_returns_ann
        S  = inputs.cov_ann
        ef = EfficientFrontier(mu, S)
        r_min, r_max = float(mu.min()), float(mu.max())
        grid = np.linspace(r_min, r_max, n_points)

        vols, rets = [], []
        for target in grid:
            ef_tmp = EfficientFrontier(mu, S)
            try:
                ef_tmp.efficient_return(target_return=target)
                perf = ef_tmp.portfolio_performance(risk_free_rate=inputs.rf_rate)
                vols.append(perf[1])  # volatility
                rets.append(perf[0])  # expected return
            except Exception:
                # some targets are infeasible
                continue

        return pd.DataFrame({"vol": vols, "ret": rets}).sort_values("vol").reset_index(drop=True)

    def max_sharpe(self, inputs: PortfolioInputs) -> Tuple[Dict[str,float], Tuple[float,float,float]]:
        ef = EfficientFrontier(inputs.exp_returns_ann, inputs.cov_ann)
        w = ef.max_sharpe(risk_free_rate=inputs.rf_rate)
        cleaned = ef.clean_weights(cutoff=1e-4)
        perf = ef.portfolio_performance(risk_free_rate=inputs.rf_rate)  # (ret, vol, sharpe)
        return cleaned, perf

    def min_volatility(self, inputs: PortfolioInputs) -> Tuple[Dict[str,float], Tuple[float,float,float]]:
        ef = EfficientFrontier(inputs.exp_returns_ann, inputs.cov_ann)
        w = ef.min_volatility()
        cleaned = ef.clean_weights(cutoff=1e-4)
        perf = ef.portfolio_performance(risk_free_rate=inputs.rf_rate)
        return cleaned, perf
