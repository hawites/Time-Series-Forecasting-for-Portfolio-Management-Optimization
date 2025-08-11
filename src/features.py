from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .config import Settings

class FeatureEngineer:
    """Cleans, merges, and derives features (returns) for all tickers."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)

    def _coerce_numeric_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure numeric for all expected price/volume columns
        cols = ["Open","High","Low","Close","Adj Close","Volume"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def merge_clean(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge per-ticker frames, pivot to wide Adj Close, clean missing."""
        df_all = []
        for df in frames:
            if "Ticker" not in df.columns:
                raise ValueError("Each DataFrame must include a 'Ticker' column.")
            # coerce numerics early
            df = self._coerce_numeric_cols(df.copy())
            df_all.append(df[["Date","Open","High","Low","Close","Adj Close","Volume","Ticker"]])

        big = pd.concat(df_all, ignore_index=True)
        # ensure datetime and sorted index
        big["Date"] = pd.to_datetime(big["Date"], errors="coerce")
        big = big.sort_values("Date").dropna(subset=["Date"])

        wide = big.pivot(index="Date", columns="Ticker", values="Adj Close").sort_index()

        # final numeric coercion on the wide matrix
        wide = wide.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        wide = wide.ffill().bfill()
        return wide

    def add_returns(self, adj_close: pd.DataFrame) -> pd.DataFrame:
        """Add simple & log returns; drop initial NA."""
        # safety: ensure numeric again if something slipped through
        adj_close = adj_close.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        adj_close = adj_close.ffill().bfill()

        pct = adj_close.pct_change().add_suffix("_ret")
        logret = np.log(adj_close / adj_close.shift(1)).add_suffix("_logret")
        out = pd.concat([adj_close, pct, logret], axis=1).dropna()
        return out

    def save(self, df: pd.DataFrame, name: str = "merged_features.csv") -> Path:
        path = self.cfg.data_processed_dir / name
        df.to_csv(path)
        return path

    def pipeline(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        wide = self.merge_clean(frames)
        feats = self.add_returns(wide)
        self.save(feats)
        return feats
