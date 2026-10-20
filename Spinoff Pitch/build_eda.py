"""Generate eda.ipynb: spin-off returns after distribution (forced selling by index funds and mandate holders)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Spinoff Pitch EDA: are spin-offs dumped at birth and rewarded later?

**The mechanism.** Holders of the parent receive spinco shares they did not choose: index funds (if the spinco is
not in their index), large-cap or sector mandates. They sell in the first days and weeks regardless of price.
Literature (Cusatis, Miles & Woolridge 1993; Greenblatt) finds spincos lag at first and outperform over the next year.

**Design.** Form 10-12B registrants 2005–2025 (SEC full-text search), mapped to *current* tickers (survivorship: dead
spincos missing), first Yahoo trading day within 18 months of the first filing (`collect.py`). Returns relative to IWM:
days 1–5, 1–20, 21–60, 61–250. 2005–15 vs 2016–.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
f=pd.read_csv("data/filers.csv",parse_dates=["first_filed"]); p=pd.read_csv("data/prices.csv",parse_dates=["date"])
A=p.pivot(index="date",columns="ticker",values="adj_close"); V=p.pivot(index="date",columns="ticker",values="volume")
iwm=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"]).pivot(index="date",columns="ticker",values="adj_close").IWM
R=np.log(A).diff()*1e4; cal=R.index; ib=(np.log(iwm).diff()*1e4).reindex(cal)
X=R.sub(ib,axis=0)
rows=[]
for r in f.itertuples():
    t=r.ticker.replace(".","-")
    if t not in A: continue
    first=A[t].first_valid_index()
    if first is None or not (r.first_filed<=first<=r.first_filed+pd.Timedelta(days=550)): continue
    i=cal.get_loc(first)
    if i+252>=len(cal): continue
    x=X[t].values
    v=V[t].values
    rows.append({"t":t,"name":r.name,"first":first,"d1_5":np.nansum(x[i+1:i+6]),"d1_20":np.nansum(x[i+1:i+21]),"d21_60":np.nansum(x[i+21:i+61]),"d61_250":np.nansum(x[i+61:i+251]),
                 "vol_wk1_vs_m3":np.nanmean(v[i:i+5])/np.nanmean(v[i+60:i+80])})
E=pd.DataFrame(rows); E["era"]=np.where(E["first"]<"2016-01-01","2005-15","2016-")
print(len(E),"spincos;", E.groupby("era").size().to_dict(), "median week-1 volume / month-3 volume:", E.vol_wk1_vs_m3.median().round(1))
for era in ["2005-15","2016-"]:
    d=E[E.era==era]; w=lambda c: d[c].clip(d[c].quantile(.02),d[c].quantile(.98))
    print(era,{c:f"{w(c).mean():+.0f} (t {w(c).mean()/w(c).sem():+.1f}) med {d[c].median():+.0f}" for c in ["d1_5","d1_20","d21_60","d61_250"]})

""")
md(r"""
**Interpretation.** No reliable pattern. Days 1–20 are negative in both eras (−271 / −303bp) but insignificant
(t ≈ −1.1) with medians of the opposite sign in 2005–15. The day 61–250 outperformance (+697bp) is not significant and
does not appear after 2016. Data quality limits it: the median volume in the first week is no higher than three
months later, so many Yahoo histories do not start at the true first regular-way trading day, and delisted spincos
are missing.

**Verdict: dead with free data.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
