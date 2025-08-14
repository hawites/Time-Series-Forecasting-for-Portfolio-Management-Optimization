import pandas as pd
from src.splits import TimeSeriesSplitter

def test_splitter_basic():
    idx = pd.date_range("2023-12-20", periods=30, freq="D")
    s = pd.Series(range(len(idx)), index=idx)
    sp = TimeSeriesSplitter(train_end="2023-12-31", test_start="2024-01-01")
    train, test = sp.split(s)
    assert train.index.max() <= pd.Timestamp("2023-12-31")
    assert test.index.min() >= pd.Timestamp("2024-01-01")
