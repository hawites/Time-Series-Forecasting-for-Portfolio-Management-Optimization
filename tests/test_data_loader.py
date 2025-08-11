import pandas as pd
from pathlib import Path
from datetime import datetime
from src.config import Settings
from src.data_loader import DataLoader

def test_load_all_list_with_existing_csvs(tmp_path, monkeypatch):
    # Arrange: create fake CSVs
    cfg = Settings(
        data_raw_dir=tmp_path / "raw",
        data_processed_dir=tmp_path / "processed",
        reports_figures_dir=tmp_path / "figs",
    )
    cfg.data_raw_dir.mkdir(parents=True, exist_ok=True)
    for t in ["TSLA","BND","SPY"]:
        df = pd.DataFrame({
            "Date":[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "Open":[1,2],"High":[2,3],"Low":[0.5,1.5],"Close":[1.5,2.5],
            "Adj Close":[1.4,2.4],"Volume":[100,200]
        })
        df.to_csv(cfg.data_raw_dir / f"{t}.csv", index=False)

    dl = DataLoader(cfg)
    frames = dl.load_all_list()
    assert len(frames) == 3
    assert all("Ticker" in f.columns for f in frames)
