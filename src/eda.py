from __future__ import annotations
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from .utils.metrics import Metrics

class EDAAnalyzer:
   
    @staticmethod
    def basic_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df[cols].describe()

    @staticmethod
    def adf_test(series: pd.Series) -> dict:
        series = series.dropna()
        stat, pval, *_ = adfuller(series, autolag="AIC")
        return {"adf_stat": float(stat), "p_value": float(pval)}

    @staticmethod
    def rolling_stats(series: pd.Series, windows=(21, 63, 252)) -> pd.DataFrame:
        out = {}
        for w in windows:
            out[f"mean_{w}"] = series.rolling(w).mean()
            out[f"std_{w}"] = series.rolling(w).std()
        return pd.DataFrame(out)

    @staticmethod
    def var_95(daily_returns: pd.Series) -> float:
        return Metrics.historical_var(daily_returns, alpha=0.95)

    @staticmethod
    def sharpe(daily_returns: pd.Series, rf_annual=0.02) -> float:
        return Metrics.sharpe_ratio(daily_returns, rf_annual)
