from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Metrics:
   
    @staticmethod
    def mae(y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        return float(mean_squared_error(y_true, y_pred, squared=False))

    @staticmethod
    def mape(y_true, y_pred) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        eps = 1e-8
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

    @staticmethod
    def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.02) -> float:
        mu_d, sd_d = daily_returns.mean(), daily_returns.std()
        if sd_d == 0:
            return float("nan")
        mu_a = mu_d * 252
        sd_a = sd_d * (252 ** 0.5)
        return float((mu_a - rf_annual) / sd_a)

    @staticmethod
    def historical_var(daily_returns: pd.Series, alpha: float = 0.95) -> float:
        return float(np.percentile(daily_returns.dropna(), (1 - alpha) * 100))
