
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yfinance as yf
from .config import Settings

class DataLoader:
    """Fetches and caches historical price data from yfinance."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.cfg.data_raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_and_cache(self, tickers: List[str] | None = None) -> Dict[str, Path]:
        
        tickers = tickers or self.cfg.tickers
        out: Dict[str, Path] = {}
        for t in tickers:
            df = yf.download(t, start=self.cfg.start, end=self.cfg.end, auto_adjust=False)
            df.reset_index(inplace=True)  # keep Date as a column
            path = self.cfg.data_raw_dir / f"{t}.csv"
            df.to_csv(path, index=False)
            out[t] = path
        return out

    def load_csv(self, ticker: str) -> pd.DataFrame:
        """Load a single ticker CSV from cache."""
        path = self.cfg.data_raw_dir / f"{ticker}.csv"
        return pd.read_csv(path, parse_dates=["Date"])

    def load_all(self, tickers: List[str] | None = None) -> Dict[str, pd.DataFrame]:
        """Return dict[ticker -> DataFrame] without Ticker column."""
        tickers = tickers or self.cfg.tickers
        return {t: self.load_csv(t) for t in tickers}

    def load_all_list(self, tickers: List[str] | None = None) -> List[pd.DataFrame]:
        """Return list of DataFrames with a Ticker column added."""
        tickers = tickers or self.cfg.tickers
        frames: List[pd.DataFrame] = []
        for t in tickers:
            df = self.load_csv(t).copy()
            df["Ticker"] = t
            frames.append(df)
        return frames
