import numpy as np
import pandas as pd
from src.backtest.backtester import Backtester, BacktestConfig

def _synth_returns(n=260):
    idx = pd.date_range("2024-08-01", periods=n, freq="B")
    df = pd.DataFrame({
        "TSLA": np.random.normal(0.001, 0.03, n),
        "BND":  np.random.normal(0.0001,0.003, n),
        "SPY":  np.random.normal(0.0006,0.01, n),
    }, index=idx)
    return df

def test_backtester_no_rebal():
    np.random.seed(0)
    R = _synth_returns()
    cfg = BacktestConfig(start="2024-08-01", end=str(R.index[-1].date()), rebalance="none", rf_annual=0.02)
    bt = Backtester(R, cfg)
    res = bt.run(strategy_weights={"TSLA":0.3,"BND":0.2,"SPY":0.5})
    assert set(res.cumrets.columns) == {"strategy","benchmark"}
    assert res.cumrets.iloc[0].min() > 0
    assert {"annual_ret","annual_vol","sharpe"}.issubset(res.stats.columns)

def test_backtester_monthly_rebal():
    np.random.seed(0)
    R = _synth_returns()
    cfg = BacktestConfig(start="2024-08-01", end=str(R.index[-1].date()), rebalance="monthly", rf_annual=0.02)
    bt = Backtester(R, cfg)
    res = bt.run(strategy_weights={"TSLA":0.3,"BND":0.2,"SPY":0.5})
    # shapes consistent
    assert len(res.daily) == len(R.loc[cfg.start:cfg.end])
