import numpy as np
import pandas as pd

from src.forecast import ARIMAForecaster, ForecastRequest

def test_arima_forecast_shapes():
    # synthetic stationary returns (AR(1))
    np.random.seed(0)
    n = 600
    phi = 0.4
    eps = np.random.normal(0, 0.01, n)
    r = [0.0]
    for t in range(1, n):
        r.append(phi * r[-1] + eps[t])
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    s = pd.Series(r, index=idx)

    train = s.iloc[:-126]
    last_price = 100.0
    last_date  = train.index[-1]

    req = ForecastRequest(steps=63, alpha=0.05, trend="n", grid_p=range(0,3), grid_d=range(0,2), grid_q=range(0,3))
    fore = ARIMAForecaster(req).fit(train)
    out = fore.forecast(train, price_train_last=last_price, last_train_date=last_date)

    assert len(out.index) == 63
    assert out.ret_mean.shape == (63,)
    assert out.px_mean.shape == (63,)
    assert np.all(np.isfinite(out.px_mean.values))
