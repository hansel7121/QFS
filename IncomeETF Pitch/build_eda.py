"""Generate eda.ipynb: ex-dividend behaviour of income / covered-call ETFs (dividend-capture demand)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# IncomeETF Pitch EDA: does dividend-capture demand distort income ETFs around ex-dates?

**The mechanism.** Covered-call and "income" ETFs (JEPI, QYLD, and especially the YieldMax single-stock funds, paying
1–3% *per distribution*) are bought by retail investors for the payout. Buyers who want the distribution bid the fund
up into the last cum day and sell after it, so the price should fall by *more* than the distribution on the ex-date
(total return below what the underlying implies).

**Design.** Abnormal total return = ETF log return (adjusted for distributions) minus a rolling 120-day beta × the
underlying's return (SPY/QQQ/IWM, or the single stock for YieldMax). Windows: 3 days before the ex-date, the ex-date,
and 3 days after. Data: Yahoo (`collect.py`). TSLA prices failed to download, so TSLY drops out.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
P = pd.read_csv("data/prices.csv", parse_dates=["date"]); A = P.pivot(index="date", columns="ticker", values="adj_close"); C = P.pivot(index="date", columns="ticker", values="close")
dv = pd.read_csv("data/dividends.csv", parse_dates=["ex_date"]); pairs = pd.read_csv("data/pairs.csv").set_index("etf").under
R = np.log(A).diff() * 1e4; cal = R.index
rows = []
for t, u in pairs.items():
    if t not in R or u not in R: continue
    x, y = R[t], R[u]
    beta = x.rolling(120, min_periods=60).cov(y) / y.rolling(120, min_periods=60).var()
    ab = x - beta.shift(1) * y
    for r in dv[dv.ticker == t].itertuples():
        i = cal.searchsorted(r.ex_date)
        if i < 130 or i + 3 >= len(cal) or cal[i] != r.ex_date: continue
        rows.append({"etf": t, "date": cal[i], "yield_bp": r.amount / C[t].iloc[i - 1] * 1e4, "pre3": ab.iloc[i - 3:i].sum(), "ex": ab.iloc[i], "post3": ab.iloc[i + 1:i + 4].sum()})
E = pd.DataFrame(rows)
E["ym"] = E.etf.isin(["TSLY", "NVDY", "CONY", "MSTY", "AMDY", "APLY", "AMZY", "GOOY", "FBY", "NFLY", "MSFO", "PLTY", "ULTY", "YMAX"])
for g, m in [("index income ETFs", ~E.ym), ("YieldMax single-stock", E.ym)]:
    d = E[m]
    print(g, "n", len(d), f"first {d.date.min().date()}", "median distribution", f"{d.yield_bp.median():.0f}bp", {c: f"{d[c].mean():+.1f} (t {d[c].mean() / d[c].sem():+.2f})" for c in ["pre3", "ex", "post3"]})
    d2 = d[d.yield_bp > d.yield_bp.quantile(0.75)]
    print("   top-quartile distributions:", {c: f"{d2[c].mean():+.1f} (t {d2[c].mean() / d2[c].sem():+.2f})" for c in ["pre3", "ex", "post3"]})
""")
md(r"""
**Interpretation.**
- **Index income ETFs: nothing** (ex-date −2.5bp, t −1.8; no pre-ex run-up).
- **YieldMax: the price over-drops on the ex-date:** −6bp abnormal (t −3.4), −17bp for the largest distributions
  (t −3.7). There is no run-up before (−4.5bp), so it is ex-day selling (capture sellers) more than a pre-ex bid.
- **Not tradable.** Capturing it means shorting at the cum-day close and covering on the ex-date: 10–15bp of spread
  on these funds plus borrow (often expensive) is about the size of the edge. The sample only starts in 2023, so there
  is no out-of-sample period.

**Verdict: real but too small and too young.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
