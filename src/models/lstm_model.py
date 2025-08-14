from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    # Only import if available (Windows+Py3.13 may lack wheels)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except Exception:  # ImportError or runtime issues
    TENSORFLOW_AVAILABLE = False

class LSTMModel:
    """Univariate LSTM forecaster for returns. Requires TensorFlow/Keras."""
    def __init__(self, lookback: int = 60, horizon: int = 1, units: int = 64, dropout: float = 0.2, seed: int = 42):
        self.lookback = lookback
        self.horizon = horizon
        self.units = units
        self.dropout = dropout
        self.seed = seed
        self.model: Optional[Sequential] = None

    def _make_windows(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(arr) - self.lookback - self.horizon + 1):
            X.append(arr[i:i+self.lookback, 0])
            y.append(arr[i+self.lookback:i+self.lookback+self.horizon, 0])
        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y).reshape(-1, self.horizon)
        return X, y

    def fit(self, train: pd.Series, epochs: int = 30, batch_size: int = 32, verbose: int = 0) -> "LSTMModel":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TF (prefer Python 3.11) to use LSTMModel.")
        np.random.seed(self.seed)
        tr = pd.Series(train).astype(float).dropna().values.reshape(-1,1)
        Xtr, ytr = self._make_windows(tr)
        self.model = Sequential([
            LSTM(self.units, input_shape=(self.lookback,1), return_sequences=False),
            Dropout(self.dropout),
            Dense(self.horizon)
        ])
        self.model.compile(loss="mse", optimizer="adam")
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(Xtr, ytr, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)
        return self

    def forecast(self, train: pd.Series, steps: int) -> np.ndarray:
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install TF (prefer Python 3.11) to use LSTMModel.")
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        history = pd.Series(train).astype(float).dropna().values.reshape(-1,1)
        preds = []
        window = history[-self.lookback:].reshape(1, self.lookback, 1)
        for _ in range(steps):
            yhat = float(self.model.predict(window, verbose=0)[0, 0])
            preds.append(yhat)
            # roll window
            new_seq = np.concatenate([window.reshape(-1,1)[1:], np.array([[yhat]])], axis=0)
            window = new_seq.reshape(1, self.lookback, 1)
        return np.array(preds, dtype=float)
