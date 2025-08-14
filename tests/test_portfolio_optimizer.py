import numpy as np
import pandas as pd
from src.portfolio.optimizer import PortfolioOptimizer, PortfolioInputs

def test_optimizer_frontier_and_points():
    np.random.seed(42)
    n = 1000
    tickers = ["TSLA","BND","SPY"]

    # synthetic daily returns with different vol levels
    rets = pd.DataFrame({
        "TSLA_ret": np.random.normal(0.001, 0.03, n),
        "BND_ret":  np.random.normal(0.0001,0.003, n),
        "SPY_ret":  np.random.normal(0.0006,0.01, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    opt = PortfolioOptimizer(rf_rate=0.02)
    exp = pd.Series({
        "TSLA": rets["TSLA_ret"].mean()*252,
        "BND":  rets["BND_ret"].mean()*252,
        "SPY":  rets["SPY_ret"].mean()*252,
    })[tickers]

    cov = opt.build_covariance(rets, tickers)

    inputs = PortfolioInputs(tickers=tickers, exp_returns_ann=exp, cov_ann=cov, rf_rate=0.02)

    # frontier
    fr = opt.efficient_frontier(inputs, n_points=10)
    assert set(["vol","ret"]).issubset(fr.columns)
    assert len(fr) > 0

    # max sharpe & min vol
    w_max, perf_max = opt.max_sharpe(inputs)
    w_min, perf_min = opt.min_volatility(inputs)

    # weights sum ~ 1
    assert abs(sum(w_max.values()) - 1.0) < 1e-6
    assert abs(sum(w_min.values()) - 1.0) < 1e-6

    # performance tuple (ret, vol, sharpe)
    assert len(perf_max) == 3 and len(perf_min) == 3
