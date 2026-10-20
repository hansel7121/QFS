"""Generate eda.ipynb: quarter-end window dressing / "leaning for the tape" (Carhart, Kaniel, Musto & Reed 2002)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Dressing Pitch EDA: do fund managers mark up their winners at quarter-end?

**The mechanism.** Carhart, Kaniel, Musto & Reed (*JF* 2002): fund managers buy their holdings, especially winners,
into the quarter-end close to lift reported performance. Prices rise on the last day and reverse the next morning. If
so, quarter-to-date winners should beat losers on the last day of a quarter (not other months) and give it back on the
first day.

**Design.** Current S&P 500 members (survivorship bias), `Overnight Pitch/data/panel.csv`. At the close of T−1, rank
on the quarter-to-date (or month-to-date) return. Outcomes: top-minus-bottom decile return on day T (close-to-close
and open-to-close), and on T+1. Returns relative to the equal-weighted universe.
""")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
p=pd.read_csv("../Overnight Pitch/data/panel.csv",parse_dates=["date"])
A=p.pivot(index="date",columns="ticker",values="adj_close"); O=p.pivot(index="date",columns="ticker",values="open"); C=p.pivot(index="date",columns="ticker",values="close")
A=A.drop(columns="SPY"); O=O[A.columns]; C=C[A.columns]
R=np.log(A).diff()*1e4; cal=A.index; ew=R.mean(1); X=R.sub(ew,axis=0)
ym=cal.to_period("M"); last=pd.Series(cal,index=cal).groupby(ym).transform("max")==pd.Series(cal,index=cal)
rows=[]
for i in np.where(last.values)[0]:
    if i<70 or i+2>=len(cal): continue
    d=cal[i]; q=d.month in (3,6,9,12)
    start=cal[i-63] if q else cal[i-21]
    sig=np.log(A.iloc[i-1]/A.loc[start])          # period-to-date return known at close T-1 (quarter or month)
    rk=sig.rank(pct=True)
    w=X.iloc[i]; n=X.iloc[i+1]
    # last-day close move vs first-day reversal; also last 30 min not available -> use open->close of last day
    idT=np.log(C.iloc[i]/O.iloc[i])*1e4 - (np.log(C.iloc[i]/O.iloc[i])*1e4).mean()
    rows.append({"date":d,"qend":q,"last_W-L":w[rk>0.9].mean()-w[rk<=0.1].mean(),"last_id_W-L":idT[rk>0.9].mean()-idT[rk<=0.1].mean(),"next_W-L":n[rk>0.9].mean()-n[rk<=0.1].mean()})
D=pd.DataFrame(rows)
for per,m in [("2005-2014",(D.date>="2005")&(D.date<"2015")),("2015-",D.date>="2015")]:
    for q in [True,False]:
        d=D[m&(D.qend==q)]
        print(per,"quarter-end" if q else "other month-end", "n",len(d), {c:f"{d[c].mean():+.1f} (t {d[c].mean()/d[c].sem():+.2f})" for c in ["last_W-L","last_id_W-L","next_W-L"]})

""")
md(r"""
**Interpretation.**
- **No markup.** Quarter-end winners do not beat losers on the last day. In 2005–14 they *lag* (−29bp, intraday
  −31bp, t −2.0); since 2015 it is zero. Other month-ends show the same lag with a next-day bounce (+25bp, t ≈ 2), the
  opposite of the dressing story and not specific to quarter-ends.

**Verdict: dead.** If anything, month-end selling of winners (rebalancing), not window dressing.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
