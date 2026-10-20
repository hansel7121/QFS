"""Moody's Baa corporate yield minus 10-year Treasury (FRED BAA10Y), daily, 2012 onward. Free and full-history
(the ICE BofA high-yield OAS on FRED now only serves the last 3 years). Output: data/baa10y.csv."""

from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"
df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y&cosd=2012-01-01", na_values=".")
df.columns = ["date", "baa10y"]
df.dropna().to_csv(DATA / "baa10y.csv", index=False)
print(len(df), df.date.min(), df.date.max())
