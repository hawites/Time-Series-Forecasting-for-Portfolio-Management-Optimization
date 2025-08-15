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
    def line_with_ci(self, mean: pd.Series, lower: pd.Series, upper: pd.Series,
                     title: str, fname: str, ylabel: str = "Value") -> Path:
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(mean.index, mean.values, label="Forecast (mean)")
        ax.fill_between(mean.index, lower.values, upper.values, alpha=0.25, label="CI band")
        ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
        ax.legend()
        path = self.out_dir / fname
        plt.tight_layout(); plt.savefig(path); plt.close()
        return path
    def efficient_frontier(self, frontier_df, maxpt, minvolpt, title, fname):
        
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9,6))
        ax.plot(frontier_df["vol"], frontier_df["ret"], label="Efficient Frontier")
        ax.scatter([maxpt[0]], [maxpt[1]], marker="*", s=180, label="Max Sharpe")
        ax.scatter([minvolpt[0]], [minvolpt[1]], marker="o", s=120, label="Min Vol")
        ax.set_xlabel("Volatility (σ)")
        ax.set_ylabel("Expected Return (μ)")
        ax.set_title(title)
        ax.legend()
        path = self.out_dir / fname
        plt.tight_layout(); plt.savefig(path); plt.close()
        return path
