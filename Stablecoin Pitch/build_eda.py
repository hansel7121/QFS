"""Generate eda.ipynb: stablecoin supply growth (new dollars entering crypto) and bitcoin returns."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Stablecoin Pitch EDA: does new Tether issuance predict bitcoin?

**The mechanism.** USDT/USDC are minted when someone wires dollars in to buy crypto. New supply is cash arriving
on exchanges, which should be spent on bitcoin over the next days (a flow), versus minting simply following past
rallies (no prediction).

**Data.** DefiLlama daily circulating supply (`collect.py`), Binance BTC daily (UTC) closes. Signals: daily log
supply growth (bp) and its z-score vs the last 90 days. Outcomes: same day, next day, next 7 days. 2020–22 vs 2023–.
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm
s=pd.read_csv("data/stables.csv",parse_dates=["date"],index_col="date")
k=pd.read_csv("../Funding Pitch/data/klines_BTCUSDT.csv",parse_dates=["open_time"])
btc=k.set_index("open_time").close.resample("1D").last()          # UTC daily close (00:00 next day)
r=np.log(btc).diff()*1e4
D=pd.DataFrame({"r":r})
for c in ["total_usd","usdt_usd"]:
    g=np.log(s[c].replace(0,np.nan)).diff()*1e4                     # bp daily supply growth (DefiLlama date = 00:00 UTC snapshot)
    D[c]=g.reindex(D.index)
D["r_next"]=D.r.shift(-1); D["r_next7"]=D.r.shift(-1).rolling(7).sum().shift(-6)
D=D[D.index>="2020-01-01"].dropna()
D["z"]=(D.usdt_usd-D.usdt_usd.rolling(90).mean())/D.usdt_usd.rolling(90).std()
for per,m in [("2020-2022",D.index<"2023"),("2023-",D.index>="2023")]:
    d=D[m].dropna()
    for sig in ["usdt_usd","total_usd","z"]:
        for y_ in ["r","r_next","r_next7"]:
            f=sm.OLS(d[y_],sm.add_constant(d[sig])).fit(cov_type="HAC",cov_kwds={"maxlags":7})
            print(f"{per} {sig:9s} -> {y_:7s}: slope {f.params[sig]:+.3f} (t {f.tvalues[sig]:+.2f})")

""")
md(r"""
**Interpretation.**
- **2020–22: nothing** (all |t| ≤ 1.3).
- **2023–: USDT growth predicts BTC:** z-score → next day +19bp per SD (t 2.5), next 7 days +57bp (t 2.5), plus
  a strong same-day link (t 3.2). Total stablecoin supply does not predict (t 0.2–0.3). It is USDT-specific (offshore
  and exchange demand).
- **Regime-only again**, like the other crypto leads: no in-sample period where it held, and DefiLlama's snapshot time
  is approximate, so a same-day overlap cannot be ruled out.

**Verdict: weak lead.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
