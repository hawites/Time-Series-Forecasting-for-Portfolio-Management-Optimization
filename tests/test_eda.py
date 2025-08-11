import numpy as np
import pandas as pd
from src.eda import EDAAnalyzer

def test_adf_and_rolling():
    s = pd.Series(np.random.normal(0, 1, 300))
    res = EDAAnalyzer.adf_test(s)
    assert "adf_stat" in res and "p_value" in res

    roll = EDAAnalyzer.rolling_stats(s, windows=(5,10))
    assert "mean_5" in roll.columns and "std_10" in roll.columns

def test_var_sharpe():
    r = pd.Series(np.random.normal(0.0005, 0.01, 252))
    assert isinstance(EDAAnalyzer.var_95(r), float)
    assert isinstance(EDAAnalyzer.sharpe(r), float)
