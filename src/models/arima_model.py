# src/models/arima_model.py
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

class ARIMAModel:
 

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        grid_p: Iterable[int] = range(0, 4),
        grid_d: Iterable[int] = range(0, 2),
        grid_q: Iterable[int] = range(0, 4),
        trend: str | None = "n",
        fit_maxiter: int = 200,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ) -> None:
        self.order = order
        self.grid_p = list(grid_p)
        self.grid_d = list(grid_d)
        self.grid_q = list(grid_q)
        self.trend = trend
        self.fit_maxiter = fit_maxiter
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._fit_res = None

    def _fit_try(self, y: pd.Series, order: Tuple[int,int,int], maxiter: int) -> Optional[object]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                res = ARIMA(
                    y, order=order, trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility
                ).fit(maxiter=maxiter)
            # Check convergence flag if available
            converged = True
            try:
                converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
            except Exception:
                pass
            if not converged:
                # one retry with more iterations
                res = ARIMA(
                    y, order=order, trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility
                ).fit(maxiter=maxiter * 2)
            return res
        except Exception:
            return None

    def select_order(self, y: pd.Series) -> Tuple[int,int,int]:
        y = pd.Series(y).astype(float).dropna()
        best_aic = np.inf
        best: Optional[Tuple[int,int,int]] = None

        for p in self.grid_p:
            for d in self.grid_d:
                for q in self.grid_q:
                    res = self._fit_try(y, (p,d,q), self.fit_maxiter)
                    if res is None or not np.isfinite(getattr(res, "aic", np.inf)):
                        continue
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best = (p,d,q)

        if best is None:
            best = (1,0,0)  # conservative fallback for returns
        self.order = best
        return best

    def fit(self, y_train: pd.Series) -> "ARIMAModel":
        y = pd.Series(y_train).astype(float).dropna()
        # Work on RangeIndex to avoid freq warnings
        y.index = pd.RangeIndex(len(y))
        if self.order is None:
            self.select_order(y)
        res = self._fit_try(y, self.order, self.fit_maxiter)
        if res is None:
            # hard fallback
            self.order = (1,0,0)
            res = self._fit_try(y, self.order, self.fit_maxiter)
            if res is None:
                raise RuntimeError("ARIMA fit failed for all attempts.")
        self._fit_res = res
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
        conf = f.conf_int(alpha=alpha).to_numpy()
        return mean, conf
