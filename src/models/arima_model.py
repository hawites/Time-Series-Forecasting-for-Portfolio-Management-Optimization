# src/models/arima_model.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
 

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        grid_p: Iterable[int] = range(0, 4),
        grid_d: Iterable[int] = range(0, 3),
        grid_q: Iterable[int] = range(0, 4),
        trend: str | None = None,  # use "n" (no trend) for returns if desired
    ) -> None:
        self.order = order
        self.grid_p = list(grid_p)
        self.grid_d = list(grid_d)
        self.grid_q = list(grid_q)
        self.trend = trend
        self._fit_res = None

    def select_order(self, y: pd.Series) -> Tuple[int, int, int]:
        best_aic = np.inf
        best: Optional[Tuple[int, int, int]] = None
        y = pd.Series(y).astype(float).dropna()

        for p in self.grid_p:
            for d in self.grid_d:
                for q in self.grid_q:
                    try:
                        # No disp kwarg; keep it minimal for broad compatibility
                        res = ARIMA(y, order=(p, d, q), trend=self.trend).fit()
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best = (p, d, q)
                    except Exception:
                        continue

        if best is None:
            best = (1, 0, 0)  # fallback
        self.order = best
        return best

    def fit(self, y_train: pd.Series) -> "ARIMAModel":
        y = pd.Series(y_train).astype(float).dropna()
        if self.order is None:
            self.select_order(y)
        # No disp kwarg here either
        self._fit_res = ARIMA(y, order=self.order, trend=self.trend).fit()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._fit_res is None:
            raise RuntimeError("Model not fitted.")
        f = self._fit_res.get_forecast(steps=steps)
        return np.asarray(f.predicted_mean)

    def forecast_with_ci(self, steps: int, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        if self._fit_res is None:
            raise RuntimeError("Model not fitted.")
        f = self._fit_res.get_forecast(steps=steps)
        mean = np.asarray(f.predicted_mean)
        conf = f.conf_int(alpha=alpha).to_numpy()  # shape (steps, 2)
        return mean, conf
