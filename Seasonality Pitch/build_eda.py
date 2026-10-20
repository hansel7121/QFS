"""Generate eda.ipynb: same-calendar-month return seasonality (Heston & Sadka 2008), out of sample."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Seasonality Pitch EDA: do stocks repeat their same-calendar-month performance?

**The anomaly.** Heston & Sadka (*JFE* 2008): a stock's return in a given calendar month predicts its return in the
same month of later years, for up to 20 years, a pattern attributed to recurring calendar flows (dividends, earnings
cycles, tax timing).

**Design.** Current S&P 500 members (survivorship), monthly 2004–2026 (`Overnight Pitch/data/panel.csv`). Signal =
average market-relative return in the same calendar month 1–5 years ago (at least 3 years). Top minus bottom decile,
equal weight, monthly. 2009–16 vs 2017–; costs 0 and 20bp a month.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
p=pd.read_csv("../Overnight Pitch/data/panel.csv",parse_dates=["date"])
A=p.pivot(index="date",columns="ticker",values="adj_close").drop(columns="SPY"); A=A.where(A>0)
M=A.resample("ME").last(); R=np.log(M).diff()*1e4
R=R.where(R.abs()<8000)
X=R.sub(R.mean(1),axis=0)
rows=[]
for k in range(60,len(R)):
    m=R.index[k]
    same=[X.iloc[k-12*j] for j in range(1,6)]                  # same calendar month, 1-5 years ago
    sig=pd.concat(same,axis=1).mean(1,skipna=True).where(pd.concat(same,axis=1).notna().sum(1)>=3)
    other=X.iloc[k-60:k].mean()                                # control: average month over the past 5 years
    r=X.iloc[k]; ok=sig.notna()&r.notna()
    if ok.sum()<100: continue
    rk=sig[ok].rank(pct=True)
    ls=r[ok][rk>0.9].mean()-r[ok][rk<=0.1].mean()
    rows.append({"m":m,"ls":ls})
D=pd.DataFrame(rows).set_index("m")
for per,mk in [("2009-2016",D.index.year<=2016),("2017-",D.index.year>=2017)]:
    x=D[mk].ls
    for c in [0,20]:
        y=x-c; print(f"{per} cost {c}bp/mo: {y.mean():+.1f}bp/mo t {y.mean()/y.std()*np.sqrt(len(y)):+.2f} ann Sharpe {y.mean()/y.std()*np.sqrt(12):+.2f} n {len(y)}")

""")
md(r"""
**Interpretation.** In large caps the seasonality spread is +11bp/month (t 0.5) in 2009–16 and −38bp (t −1.1) since
2017. Nothing to trade, before or after costs.

**Verdict: dead in large caps.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
