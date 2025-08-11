import pandas as pd
from src.config import Settings
from src.features import FeatureEngineer

def test_pipeline_generates_returns(tmp_path):
    cfg = Settings(
        data_raw_dir=tmp_path / "raw",
        data_processed_dir=tmp_path / "processed",
        reports_figures_dir=tmp_path / "figs",
    )
    fe = FeatureEngineer(cfg)
    a = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "Adj Close":[10, 10.5, 10.0], "Ticker":"TSLA"
    })
    b = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "Adj Close":[20, 20.2, 20.3], "Ticker":"BND"
    })
    c = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "Adj Close":[30, 29.5, 30.0], "Ticker":"SPY"
    })
    for df in (a,b,c):
        # add required cols minimally
        df["Open"]=df["High"]=df["Low"]=df["Adj Close"]
        df["Close"]=df["Adj Close"]
        df["Volume"]=100

    out = fe.pipeline([a,b,c])
    assert "TSLA" in out.columns and "TSLA_ret" in out.columns and "TSLA_logret" in out.columns
    assert (cfg.data_processed_dir / "merged_features.csv").exists()
