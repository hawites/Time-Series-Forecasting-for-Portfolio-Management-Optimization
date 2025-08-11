from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from .config import Settings

class FeatureEngineer:
    
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)

    def merge_clean(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
       
        df_all = []
        for df in frames:
            t = df.get("Ticker", None)
            if t is None:
                # infer from filename-like columns if needed; else require user to add
                raise ValueError("Each DataFrame must include a 'Ticker' column.")
            df_all.append(df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]])
        big = pd.concat(df_all, ignore_index=True)
        wide = big.pivot(index="Date", columns="Ticker", values="Adj Close").sort_index()
        wide = wide.ffill().bfill()
        return wide

    def add_returns(self, adj_close: pd.DataFrame) -> pd.DataFrame:
       
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
