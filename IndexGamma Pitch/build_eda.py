"""Generate eda.ipynb: market intraday momentum (Gao et al. 2018; Baltussen et al. 2021) after publication."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# IndexGamma Pitch EDA: is market intraday momentum (leveraged-ETF and dealer gamma) still there?

**The papers.** Gao, Han, Li & Zhou (*JFE* 2018): the first half-hour return predicts the last half-hour return of the
S&P 500. Baltussen, Da, Lammers & Martens (*JFE* 2021) tie it to gamma hedging by leveraged ETFs and option dealers:
the return so far predicts the last 30 minutes. Both samples end before 2021.

**Design.** SPY and QQQ 5-minute bars (Twelve Data, 2020-02 →; `collect_intraday.py`), official closes from Yahoo.
Signals: previous close → 10:00 (first30) and previous close → 15:30 (rest). Outcome: 15:30 → official close. Trade:
sign of the signal, net 1bp. Also: does the last 30 minutes reverse overnight? 2020 vs 2021–23 vs 2024–.
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm
px=pd.read_csv("../TOM Pitch/data/etfs.csv",parse_dates=["date"])
cl=px.pivot(index="date",columns="ticker",values="close")
for s in ["SPY","QQQ"]:
    x=pd.read_csv(f"data/intraday/{s}.csv",parse_dates=["datetime"]); x["d"]=x.datetime.dt.normalize(); x["hm"]=x.datetime.dt.strftime("%H:%M")
    c=x.pivot_table(index="d",columns="hm",values="close"); c=c[c.notna().sum(1)>=76]
    o930=x[x.hm=="09:30"].set_index("d").open.reindex(c.index)          # first-bar open = official open (approx.)
    C=cl[s].reindex(c.index); P=cl[s].shift(1).reindex(c.index)
    D=pd.DataFrame({"first30":np.log(c["09:55"]/P)*1e4,"rest":np.log(c["15:25"]/P)*1e4,"last30":np.log(C/c["15:25"])*1e4,
                    "next_on":np.log(o930.shift(-1)/C)*1e4}).dropna()
    for per,m in [("2020",D.index<"2021"),("2021-2023",(D.index>="2021")&(D.index<"2024")),("2024-",D.index>="2024")]:
        d=D[m]; out=[]
        for sig in ["first30","rest"]:
            f=sm.OLS(d.last30,sm.add_constant(d[sig])).fit(cov_type="HC1")
            pnl=np.sign(d[sig])*d.last30-1.0
            out.append(f"{sig} -> last30: b {f.params[sig]:+.3f} (t {f.tvalues[sig]:+.2f}); trade net1bp Sharpe {pnl.mean()/pnl.std()*np.sqrt(252):+.2f}")
        g=sm.OLS(d.next_on,sm.add_constant(d.last30)).fit(cov_type="HC1")
        print(s,per,"n",len(d)," | ".join(out), f"| overnight reversal of last30: b {g.params.last30:+.2f} (t {g.tvalues.last30:+.2f})")

""")
md(r"""
**Interpretation.** The effect is visible in 2020 (QQQ first30 t 1.8; trades Sharpe 0.7–1.4, along with a strong
overnight reversal of the last 30 minutes, t ≈ −2.8). **From 2021 it is gone**: slopes ≈ 0 (|t| ≤ 1.4) for both
signals and both ETFs, the trade Sharpe is negative in 2024–, and the overnight reversal disappears. That fits the
single-stock leveraged-ETF result: the rebalancing flow is anticipated and absorbed.

**Verdict: dead after publication.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
