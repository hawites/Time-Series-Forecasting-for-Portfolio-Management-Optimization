import numpy as np
import pandas as pd
from src.utils.metrics import Metrics

def test_basic_metrics():
    y_true = [1,2,3]
    y_pred = [1,2,2.5]
    assert Metrics.mae(y_true, y_pred) > 0
    assert Metrics.rmse(y_true, y_pred) > 0
    assert Metrics.mape(y_true, y_pred) >= 0

def test_sharpe_and_var():
    rs = pd.Series(np.random.normal(0.0005, 0.01, 500))
    assert isinstance(Metrics.sharpe_ratio(rs, 0.02), float)
    assert isinstance(Metrics.historical_var(rs, 0.95), float)
