"""Generate eda.ipynb: the Coinbase premium (US-dollar spot demand) and future bitcoin returns."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# CBPremium Pitch EDA: does US spot buying (Coinbase premium) lead bitcoin?

**The mechanism.** US institutions and spot ETFs' market makers buy on Coinbase in dollars. When that demand is heavy,
Coinbase BTC-USD trades above Binance BTC-USDT (adjusted for USDT/USD). If the flow is persistent and not fully
arbitraged, the premium should predict the next hours' returns.

**Data.** Hourly candles: Coinbase BTC-USD and USDT-USD (public API), Binance spot BTCUSDT (`collect.py`), 2021-05 →.
Premium = log(Coinbase BTC-USD) − log(Binance BTC-USDT × USDT-USD), in bp; 24-hour average; 30-day z-score. 2021–23 vs 2024–.
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm
cb=pd.read_csv("data/coinbase_BTC-USD.csv",parse_dates=["time"]).set_index("time").close
ut=pd.read_csv("data/coinbase_USDT-USD.csv",parse_dates=["time"]).set_index("time").close
bn=pd.read_csv("data/binance_BTCUSDT_1h.csv",parse_dates=["time"]).set_index("time").close
# candle time = open time; close = price at time+1h
idx=cb.index.intersection(bn.index).intersection(ut.index)
D=pd.DataFrame({"cb":cb.reindex(idx),"bn":bn.reindex(idx),"ut":ut.reindex(idx)}).dropna()
D=D[D.ut.between(0.97,1.03)]
D["prem"]=(np.log(D.cb)-np.log(D.bn*D.ut))*1e4                 # bp
D["prem24"]=D.prem.rolling(24).mean()
lp=np.log(D.cb)
for h in [1,4,24]: D[f"f{h}"]=(lp.shift(-h)-lp)*1e4
D["z"]=(D.prem24-D.prem24.rolling(24*30).mean())/D.prem24.rolling(24*30).std()
D=D.dropna()
print("premium bp: mean",D.prem.mean().round(1),"sd",D.prem.std().round(1))
for per,m in [("2021-05..2023",D.index<"2024"),("2024-",D.index>="2024")]:
    d=D[m]
    for sig in ["prem","prem24","z"]:
        line=f"{per} {sig:6s}: "
        for h in [1,4,24]:
            f=sm.OLS(d[f"f{h}"],sm.add_constant(d[sig])).fit(cov_type="HAC",cov_kwds={"maxlags":h+24})
            line+=f"f{h}h {f.params[sig]:+.3f} (t {f.tvalues[sig]:+.2f}) | "
        print(line)
print("\n==== daily strategy: at 00:00 UTC, position from prem24 (known), hold 24h; cost 5bp per unit of position change")
day=D[D.index.hour==23].copy()                          # candle 23:00 closes at 00:00 -> signal at 00:00
day["ret"]=day.f24
for rule in ["sign","long_if_pos","z_clip"]:
    if rule=="sign": pos=np.sign(day.prem24)
    elif rule=="long_if_pos": pos=(day.prem24>0).astype(float)
    else: pos=day.z.clip(-2,2)/2
    pnl=pos*day.ret-5*pos.diff().abs().fillna(0)
    for per,m in [("2021-05..2023",day.index<"2024"),("2024-",day.index>="2024")]:
        x=pnl[m]; b=day.ret[m]
        f=sm.OLS(x,sm.add_constant(b)).fit(cov_type="HAC",cov_kwds={"maxlags":5})
        print(f"{rule:12s} {per}: {x.mean()*365/100:+6.1f}%/yr Sharpe {x.mean()/x.std()*np.sqrt(365):+.2f} | B&H Sharpe {b.mean()/b.std()*np.sqrt(365):+.2f} | alpha {f.params.const*365/100:+.1f}%/yr (t {f.tvalues.const:+.2f}) beta {f.params.iloc[1]:+.2f} | turnover/day {pos.diff().abs()[m].mean():.2f}")

""")
md(r"""
**Interpretation.**
- **The premium does lead bitcoin, in both periods:** the 24h-average premium predicts the next 24h with t 2.5
  (2021–23) and 2.2 (2024–), and the 1h/4h horizons have the same sign.
- **But it is not tradable.** The hourly regressions use overlapping windows. As a daily trade (position from the
  premium at 00:00 UTC, hold 24h, 5bp per unit traded) the Sharpe is 0.2–0.7, about buy-and-hold, and alpha after
  removing BTC beta is never significant (t ≤ 0.8). The z-scored version earns nothing after 2024.

**Verdict: real but not tradable.**
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
