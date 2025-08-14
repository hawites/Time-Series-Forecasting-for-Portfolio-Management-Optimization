import numpy as np
import pandas as pd
from src.models.arima_model import ARIMAModel
from src.utils.metrics import Metrics

def test_arima_on_ar1():
    np.random.seed(0)
    n = 300
    phi = 0.7
    eps = np.random.normal(0, 1, n)
    y = [0.0]
    for t in range(1, n):
        y.append(phi * y[-1] + eps[t])
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    s = pd.Series(y, index=idx)

    train = s.iloc[:250]
    test  = s.iloc[250:]

    model = ARIMAModel(grid_p=range(0,3), grid_d=range(0,2), grid_q=range(0,3)).fit(train)
    preds = model.forecast(len(test))
    assert len(preds) == len(test)
    # just sanity: metric is finite
    assert np.isfinite(Metrics.rmse(test.values, preds))
