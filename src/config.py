from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class Settings:
    """Immutable project configuration."""
    start: str = "2015-07-01"
    end: str = "2025-07-31"
    tickers: List[str] = field(default_factory=lambda: ["TSLA", "BND", "SPY"])
    risk_free_rate: float = 0.02  # annualized
    seed: int = 42

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    reports_figures_dir: Path = Path("reports/figures")
