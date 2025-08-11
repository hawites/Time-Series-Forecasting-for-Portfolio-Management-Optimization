import pandas as pd
from src.utils.plotting import Plotter

def test_plot_saves_files(tmp_path):
    pl = Plotter(tmp_path)
    df = pd.DataFrame({"A":[1,2,3]}, index=pd.date_range("2024-01-01", periods=3))
    p1 = pl.line(df, ["A"], "Title", "line.png")
    p2 = pl.series(df["A"], "Title", "series.png")
    assert p1.exists() and p2.exists()
