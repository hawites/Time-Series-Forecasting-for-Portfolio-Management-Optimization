
from __future__ import annotations
import numpy as np
import pandas as pd

class Metrics:
    

    @staticmethod
    def _to_np(a):
        return np.asarray(a, dtype=float)

    @staticmethod
    def mae(y_true, y_pred) -> float:
        y_true = Metrics._to_np(y_true)
        y_pred = Metrics._to_np(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        y_true = Metrics._to_np(y_true)
        y_pred = Metrics._to_np(y_pred)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true, y_pred) -> float:
        y_true = Metrics._to_np(y_true)
        y_pred = Metrics._to_np(y_pred)
        eps = 1e-8
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

    @staticmethod
    def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.02) -> float:
        daily_returns = pd.Series(daily_returns).dropna()
        mu_d = daily_returns.mean()
        sd_d = daily_returns.std()
        if sd_d == 0 or np.isnan(sd_d):
            return float("nan")
        mu_a = mu_d * 252
        sd_a = sd_d * np.sqrt(252)
        return float((mu_a - rf_annual) / sd_a)

    @staticmethod
    def historical_var(daily_returns: pd.Series, alpha: float = 0.95) -> float:
        daily_returns = pd.Series(daily_returns).dropna()
        return float(np.percentile(daily_returns, (1 - alpha) * 100))
