from src.config import Settings

def test_settings_defaults():
    cfg = Settings()
    assert "TSLA" in cfg.tickers and "SPY" in cfg.tickers
    assert cfg.data_raw_dir.name == "raw"
    assert cfg.risk_free_rate == 0.02
