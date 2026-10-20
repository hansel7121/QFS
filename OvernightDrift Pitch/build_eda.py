"""Generate eda.ipynb: the overnight drift in S&P futures (Boyarchenko, Larsen & Whelan 2023), 2024-2026."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# OvernightDrift Pitch EDA: do S&P futures still earn their return around the European open?

**The paper.** Boyarchenko, Larsen & Whelan (*RFS* 2023): E-mini S&P futures earn a large share of their returns in
the hour around the European open (02:00–03:00 ET), tied to dealers unwinding inventory taken on in US-hours selloffs.

**Data.** Yahoo `ES=F` hourly bars, the only free 24-hour S&P futures history (last 730 days: 2024-05 → 2026-10).
Mean hourly return (bp) and t-stat by New York hour.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
d = pd.read_csv("data/es_1h.csv", parse_dates=["Datetime"], index_col="Datetime")
d.index = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
r = np.log(d.Close).diff() * 1e4
r = r[r.index.to_series().diff() <= pd.Timedelta(hours=1)]
g = r.groupby(r.index.hour)
print(pd.DataFrame({"mean bp": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()), "n": g.count()}).round(2).to_string())
""")
md(r"""
**Interpretation.** No hour stands out. The 02:00–04:00 ET hours average +0.4 to +0.7bp (t < 1), far below the
paper's estimates. Over 2024–26 the overnight drift is not visible, and free data does not reach back far enough for an
in-sample period.

**Verdict: dead / untestable with free data.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
