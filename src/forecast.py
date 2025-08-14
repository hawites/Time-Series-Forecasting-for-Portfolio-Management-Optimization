from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .models.arima_model import ARIMAModel

@dataclass(frozen=True)
class ForecastRequest:
    steps: int = 126                 # ~6 months of trading days
    alpha: float = 0.05              # 95% CI
    trend: Optional[str] = "n"       # no constant for returns
    grid_p: range = range(0, 4)
    grid_d: range = range(0, 2)
    grid_q: range = range(0, 4)

@dataclass(frozen=True)
class ForecastResult:
    # future date index
    index: pd.DatetimeIndex
    # returns forecast (mean / CI)
    ret_mean: pd.Series
    ret_lower: pd.Series
    ret_upper: pd.Series
    # reconstructed prices (mean / CI)
    px_mean: pd.Series
    px_lower: pd.Series
    px_upper: pd.Series
    # the (p,d,q) order used
    order: Tuple[int, int, int]

class ARIMAForecaster:
   
    def __init__(self, req: ForecastRequest) -> None:
        self.req = req
        self.model = ARIMAModel(
            order=None,
            grid_p=req.grid_p,
            grid_d=req.grid_d,
            grid_q=req.grid_q,
            trend=req.trend,
        )
        self.fitted = False

    @staticmethod
    def _range_index(series: pd.Series) -> pd.Series:
        s = pd.Series(series).astype(float).dropna()
        s.index = pd.RangeIndex(len(s))
        return s

    @staticmethod
    def _future_bdays(start_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
        # Next business day forward for `steps` days
        return pd.bdate_range(start_date + pd.offsets.BDay(1), periods=steps, freq="B")

    @staticmethod
    def _reconstruct_prices(last_price: float, ret_path: np.ndarray) -> np.ndarray:
        return float(last_price) * np.cumprod(1.0 + np.asarray(ret_path, dtype=float))

    def fit(self, ret_train: pd.Series) -> "ARIMAForecaster":
        y = self._range_index(ret_train)
        self.model.fit(y)
        self.fitted = True
        return self

    def forecast(self, ret_train: pd.Series, price_train_last: float,
                 last_train_date: pd.Timestamp,
                 steps: Optional[int] = None,
                 alpha: Optional[float] = None) -> ForecastResult:
        if not self.fitted:
            self.fit(ret_train)

        steps = steps or self.req.steps
        alpha = alpha or self.req.alpha

        # returns forecast (integer index)
        mean, conf = self.model.forecast_with_ci(steps=steps, alpha=alpha)
        ret_mean = pd.Series(mean)
        # conf is shape (steps, 2): [lower, upper]
        ret_lower = pd.Series(conf[:, 0])
        ret_upper = pd.Series(conf[:, 1])

        # attach real future business dates
        idx = self._future_bdays(last_train_date, steps)
        ret_mean.index = idx
        ret_lower.index = idx
        ret_upper.index = idx

        # reconstruct price paths
        px_mean = pd.Series(self._reconstruct_prices(price_train_last, ret_mean.values), index=idx)
        px_low  = pd.Series(self._reconstruct_prices(price_train_last, ret_lower.values), index=idx)
        px_up   = pd.Series(self._reconstruct_prices(price_train_last, ret_upper.values), index=idx)

        return ForecastResult(
            index=idx,
            ret_mean=ret_mean, ret_lower=ret_lower, ret_upper=ret_upper,
            px_mean=px_mean, px_lower=px_low, px_upper=px_up,
            order=self.model.order,
        )
