"""Generate eda.ipynb: Japanese passive funds' dividend-reinvestment futures buying on the last cum-dividend day (March / September)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# JapanDiv Pitch EDA: do index funds' dividend-reinvestment buys lift the market on the last cum-dividend day?

**The mechanism.** Most Japanese companies pay dividends to holders of record at the end of March and September. On
the ex-date the index drops by the dividend, but an index fund's NAV does not: the dividend becomes a receivable that
is only paid in June / December. To stay fully invested, Nikkei/TOPIX funds buy index futures equal to the expected
dividend at the **close of the last cum-dividend day** (Japanese press: 配当再投資の先物買い, about ¥1tn in March). That
is a known-date, mechanical, price-insensitive buy, and it grew with passive assets (BOJ / GPIF era, 2013–).

**Dates.** Record date = the month's last business day; last cum day = 3 business days before it (T+2 settlement
since 2019-07-16; 4 before that under T+3). Returns: close-to-close and open-to-close on the cum day, overnight into
the ex-date, and the 3 days after. Index returns on the ex-date include the dividend drop (price index), so ETFs
(1306.T TOPIX, 1321.T Nikkei) are shown too.

**Data.** Yahoo daily (`collect_prices.py`).
""")
code(r"""
import pandas as pd, numpy as np
d=pd.read_csv("data/prices.csv",parse_dates=["date"])
O=d.pivot(index="date",columns="ticker",values="open"); C=d.pivot(index="date",columns="ticker",values="close")
out={}
for tk in ["^N225","1306.T","1321.T"]:
    n=C[tk].dropna(); o=O[tk].reindex(n.index); cal=n.index
    n=n[n>0]; cal=n.index; o=o.reindex(cal)
    cc=np.log(n).diff()*1e4; on=np.log(o/n.shift(1))*1e4; idr=np.log(n/o)*1e4
    rows=[]
    for y in range(1996,2027):
        for m in range(1,13):
            days=cal[(cal.year==y)&(cal.month==m)]
            if len(days)<5: continue
            k=2 if pd.Timestamp(y,m,1)>=pd.Timestamp("2019-07-16") else 3
            cum=days[-k-1]; i=cal.get_loc(cum)
            if i+5>=len(cal): continue
            rows.append({"tk":tk,"y":y,"m":m,"div":m in (3,9),"cum_cc":cc.iloc[i],"cum_id":idr.iloc[i],"ex_on":on.iloc[i+1],"ex_id":idr.iloc[i+1],"post3":cc.iloc[i+2:i+5].sum()})
    out[tk]=pd.DataFrame(rows)
for tk,R in out.items():
    for per,m in [("1996-2012",R.y<=2012),("2013-",R.y>=2013)]:
        d_=R[m&R["div"]]; o_=R[m&~R["div"]]
        print(tk,per,"Mar/Sep n",len(d_), {c:f"{d_[c].mean():+.0f} (t {d_[c].mean()/d_[c].sem():+.1f}) vs {o_[c].mean():+.0f}" for c in ["cum_cc","cum_id","ex_on","ex_id","post3"]})
R=out["^N225"]; R2=out["1321.T"]
print("\nN225 cum-day close-to-close by year, Mar / Sep:")
print(R[R["div"]&(R.y>=2008)].pivot(index="y",columns="m",values="cum_cc").round(0).T.to_string())

""")
md(r"""
**Interpretation.**
- **The cum-day pop exists since 2013:** Nikkei +57bp (t 2.3), TOPIX ETF +71bp (t 3.1), Nikkei ETF +41bp (t 1.9) on
  the last cum day of March/September, against +7–11bp in other months. Much of it is intraday (open → close +35–39bp),
  consistent with the close buy.
- **Before 2013 it is absent in the Nikkei** (+9bp, t 0.4), consistent with smaller passive assets, but that means
  there is no in-sample period to fit on and an out-of-sample period to test on. 2013– is simply where it exists.
- **It looks competed away recently.** The last four events (Mar/Sep 2025, Mar/Sep 2026) are all negative
  (−43 to −88bp). The run-up has moved earlier (pre-3-day +53bp) and a −80bp give-back follows over the next 3 days,
  the signature of front-running.
- 2 events a year: per-year Sharpe ≈ 0.6 at best on 28 events.

**Verdict: weak candidate, not promising enough.** The mechanism is real and visible, but the sample is small, the
split is not a true out-of-sample test, and the latest events reverse.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
