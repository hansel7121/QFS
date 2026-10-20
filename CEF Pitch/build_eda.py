"""Generate eda.ipynb: year-end tax-loss selling in closed-end funds (discount widening in December, January reversal)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# CEF Pitch EDA: does December tax-loss selling widen CEF discounts, and does January pay it back?

**The mechanism.** Retail holders sell losing closed-end funds before December 31 to book capital losses. CEF
shares cannot be redeemed at NAV, so the selling widens the discount, and it should narrow in January when the
selling stops. NAV gives a fair-value benchmark. Anchor: Brauer & Chang (1990, *RFS*), January effect in CEF prices but
not NAVs.

**Design (all known before entry).** Tax-loss proxy = price total return from January 1 to December 15. Funds are
sorted into quintiles each year (Q1 = biggest losers). Entry at the close of the last trading day of December, exit
at the close of January 31. Outcomes:
* **ret**: price total return, bp.
* **dret**: price return minus NAV return (the discount-driven part, what a NAV-hedged position would earn), bp.
* **ddisc**: discount change, pp.
* **Dec widening**: discount change Nov 30 → Dec 31, as the mechanism check.

L−W = losers minus winners. Placebo: the same at June/July. In-sample 2004–15, out-of-sample 2016–25.

**Data.** `collect_cef.py`: CEFConnect universe (today's funds, so it has **survivorship bias**), Yahoo prices and NAVs
(X…X tickers).
""")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
p=pd.read_csv("data/prices.csv",parse_dates=["date"])
px=p.pivot(index="date",columns="ticker",values="adj_close"); cl=p.pivot(index="date",columns="ticker",values="close"); nav=p.pivot(index="date",columns="ticker",values="nav")
disc=(cl/nav-1).where(lambda d:d.abs()<0.6)*100
cal=px.index
def at(d):
    i=cal.searchsorted(pd.Timestamp(d),side="right")-1
    return cal[i] if i>=0 else None
rows=[]
for y in range(2004,2026):
    for mon in ["Dec","Jun(placebo)"]:
        if mon=="Dec": s0,s,e,x,b=at(f"{y}-01-01"),at(f"{y}-12-15"),at(f"{y}-12-31"),at(f"{y+1}-01-31"),at(f"{y}-11-30")
        else: s0,s,e,x,b=at(f"{y-1}-07-01"),at(f"{y}-06-15"),at(f"{y}-06-30"),at(f"{y}-07-31"),at(f"{y}-05-31")
        if s0 is None or x is None or pd.Timestamp(f"{y+1}-01-31" if mon=="Dec" else f"{y}-07-31")>cal[-1]: continue
        ytd=px.loc[s]/px.loc[s0]-1
        ret=px.loc[x]/px.loc[e]-1; navret=nav.loc[x]/nav.loc[e]-1
        df=pd.DataFrame({"ytd":ytd,"ret":ret*1e4,"dret":(ret-navret)*1e4,"ddisc":disc.loc[x]-disc.loc[e],"ddec":disc.loc[e]-disc.loc[b]}).dropna()
        df=df[df.ytd.abs()<1]
        if len(df)<50: continue
        df["q"]=pd.qcut(df.ytd.rank(method="first"),5,labels=False)+1
        g=df.groupby("q")[["ret","dret","ddisc","ddec"]].mean()
        rows.append({"year":y,"mon":mon,"n":len(df),"L-W ret":g.ret[1]-g.ret[5],"L-W dret":g.dret[1]-g.dret[5],"L-W ddisc":g.ddisc[1]-g.ddisc[5],"L-W Dec widening":g.ddec[1]-g.ddec[5],"L ret":g.ret[1],"all ret":df.ret.mean()})
R=pd.DataFrame(rows)
for mon in ["Dec","Jun(placebo)"]:
    for per,m in [("2004-15",R.year<=2015),("2016-25",R.year>=2016)]:
        d=R[(R.mon==mon)&m]
        print(mon,per,"n yrs",len(d)); print("   ",{c:f"{d[c].mean():+.1f} (t {d[c].mean()/d[c].std()*np.sqrt(len(d)):+.1f})" for c in ["L-W ret","L-W dret","L-W ddisc","L-W Dec widening","L ret","all ret"]})
print(R[R.mon=="Dec"][["year","n","L-W ret","L-W ddisc","L-W Dec widening"]].round(1).to_string(index=False))

""")
md(r"""
**Interpretation.**
- **The mechanism is there and persists.** Losers' discounts widen more than winners' in December: −1.5pp
  (t −3.5) in 2004–15 and −0.8pp (t −3.0) in 2016–25. That is tax-loss selling hitting CEF shares and not NAVs.
- **The January payoff has shrunk.** Losers' discounts narrow more in January: +1.9pp (t 2.7) in-sample, then
  +0.5pp (t 1.7) out of sample. The NAV-hedged return (L−W dret) falls from +203bp (t 2.6) to +56bp (t 1.8, only
  10 Januaries).
- **Raw price returns are noise** (L−W ±5–14% swings by year): NAV moves dominate, so it needs a NAV hedge
  (category ETF), which adds cost and basis risk.
- **Costs.** CEF losers are small and illiquid. A round trip of 20–40bp plus the hedge eats most of the remaining
  56bp. One trade a year gives 10 out-of-sample observations.
- The placebo is clean (June/July L−W dret ≈ 0).

**Verdict: real mechanism, not tradable enough out of sample.** Close to the edge, but fails the t ≥ 2 / Sharpe bar
after costs, with survivorship bias that likely flatters it.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
