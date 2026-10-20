"""Generate eda.ipynb: December tax-loss selling and the January rebound in large-cap US stocks."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# TaxLoss Pitch EDA: do year-to-date losers get dumped in late December and bounce in January?

**The mechanism.** Taxable investors sell losers before December 31 to realise losses (forced by the calendar), then
the selling stops (Roll 1983; Reinganum 1983). Losers should lag winners in late December and rebound in early January.

**Design.** Current S&P 500 members (survivorship), `Overnight Pitch/data/panel.csv`. YTD return at December 15;
bottom-minus-top decile, relative to the equal-weighted universe, over Dec 16–24, Dec 24–31, Jan 1–15, Jan 15–31.
2005–14 vs 2015–.
""")
code(r"""
import pandas as pd, numpy as np
p=pd.read_csv("../Overnight Pitch/data/panel.csv",parse_dates=["date"])
A=p.pivot(index="date",columns="ticker",values="adj_close").drop(columns="SPY")
R=np.log(A).diff()*1e4; cal=A.index; X=R.sub(R.mean(1),axis=0)
at=lambda d: cal[cal.searchsorted(pd.Timestamp(d),side="right")-1]
rows=[]
for y in range(2005,2026):
    s0,s1=at(f"{y-1}-12-31"),at(f"{y}-12-15")
    ytd=np.log(A.loc[s1]/A.loc[s0]); rk=ytd.rank(pct=True)
    L,W=rk<=0.1,rk>0.9
    def win(a,b):
        i0,i1=cal.get_loc(at(a)),cal.get_loc(at(b)); x=X.iloc[i0+1:i1+1].sum(min_count=1); return x[L].mean()-x[W].mean()
    rows.append({"y":y,"Dec16-24":win(f"{y}-12-15",f"{y}-12-24"),"Dec24-31":win(f"{y}-12-24",f"{y}-12-31"),"Jan1-15":win(f"{y}-12-31",f"{y+1}-01-15"),"Jan15-31":win(f"{y+1}-01-15",f"{y+1}-01-31")})
D=pd.DataFrame(rows)
for per,m in [("2005-2014",D.y<=2014),("2015-",D.y>=2015)]:
    d=D[m]; print(per,{c:f"{d[c].mean():+.0f} (t {d[c].mean()/d[c].sem():+.1f})" for c in ["Dec16-24","Dec24-31","Jan1-15","Jan15-31"]})
print(D.set_index("y")[["Dec24-31","Jan1-15"]].round(0).T.to_string())

""")
md(r"""
**Interpretation.** No window is significant in either era (|t| ≤ 1.5), and the year-by-year January numbers swing
from −7% to +9%. In large caps, tax-loss pressure is lost in the noise.

**Verdict: dead in large caps** (the CEF version showed the mechanism but too small to trade).
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
