import numpy as np
import pandas as pd
import pytest

from src.models.lstm_model import LSTMModel, TENSORFLOW_AVAILABLE

@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
def test_lstm_shapes():
    # simple sine wave returns
    idx = pd.date_range("2021-01-01", periods=400, freq="D")
    s = pd.Series(np.sin(np.linspace(0, 40, 400))/100.0, index=idx)
    train = s.iloc[:300]
    test  = s.iloc[300:]

    m = LSTMModel(lookback=30, units=16, dropout=0.1)
    m.fit(train, epochs=5, batch_size=16, verbose=0)
    preds = m.forecast(train, steps=len(test))
    assert len(preds) == len(test)
