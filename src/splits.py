from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

@dataclass
class TimeSeriesSplitter:
    """Chronological train/test splitter by date."""
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"

    def split(self, s: pd.Series) -> tuple[pd.Series, pd.Series]:
        s = s.sort_index()
        train = s.loc[: self.train_end].dropna()
        test  = s.loc[self.test_start :].dropna()
        return train, test
