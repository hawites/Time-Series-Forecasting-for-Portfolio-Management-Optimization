from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def line(self, df: pd.DataFrame, cols: list[str], title: str, fname: str) -> Path:
        ax = df[cols].plot(figsize=(10, 5))
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        path = self.out_dir / fname
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def series(self, s: pd.Series, title: str, fname: str) -> Path:
        ax = s.plot(figsize=(10, 4))
        ax.set_title(title)
        ax.set_xlabel("Date")
        path = self.out_dir / fname
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path
