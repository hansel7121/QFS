"""Generate eda.ipynb: price behaviour around perpetual-futures funding settlements (Binance, 2020-2026)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Funding Pitch EDA: do traders close positions into funding settlements?

**The mechanism.** Perpetual futures charge funding every 8 hours (00:00/08:00/16:00 UTC on Binance). When funding is
very positive, longs pay shorts. A long who exits just before the settlement avoids the payment, then can re-enter just
after. That predicts selling pressure just before $T$ and a rebound just after, scaling with the funding rate.

**Signal (no look-ahead).** `fr_prev` = the rate settled at the previous settlement (fully known 8 hours before $T$).
The rate applied at $T$ itself is only approximately known in advance (Binance's live predicted rate), so it is shown
as a secondary check only.

**Data.** `collect_binance.py`: 5-minute klines and funding history for BTC, ETH, SOL, XRP, DOGE USDT perpetuals
(data.binance.vision). In-sample 2020–2023, out-of-sample 2024–. Cost: maker about 2bp per side (4bp round trip);
taker about 10bp round trip.
""")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
res = []
for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]:
    k = pd.read_csv(f"data/klines_{s}.csv", parse_dates=["open_time"]).set_index("open_time"); lc = np.log(k.close)
    f = pd.read_csv(f"data/funding_{s}.csv", parse_dates=["calc_time"]); f["T"] = f.calc_time.dt.round("5min"); f = f.set_index("T")
    px = lambda t: lc.reindex(t - pd.Timedelta(minutes=5)).values         # close of the bar ending at t
    T = f.index; d = pd.DataFrame(index=T); d["fr"] = f.last_funding_rate.values * 1e4; d["fr_prev"] = d.fr.shift(1)
    for a, b, nm in [(-240, -60, "placebo -4h..-1h"), (-60, 0, "pre60"), (-10, 0, "pre10"), (0, 10, "post10"), (0, 30, "post30"), (0, 60, "post60"), (0, 120, "post120")]:
        d[nm] = (px(T + pd.Timedelta(minutes=b)) - px(T + pd.Timedelta(minutes=a))) * 1e4
    d["sym"] = s; res.append(d.dropna())
D = pd.concat(res); D["is"] = D.index < "2024-01-01"
print(D.groupby("sym").fr.describe()[["mean", "50%", "std", "max"]].round(2).to_string())
rows = []
for sig in ["fr_prev", "fr"]:
    for w in ["placebo -4h..-1h", "pre60", "pre10", "post10", "post60", "post120"]:
        for per, m in [("in-sample", D["is"]), ("out-of-sample", ~D["is"])]:
            d = D[m]; f_ = sm.OLS(d[w], sm.add_constant(d[sig].clip(-20, 20))).fit(cov_type="cluster", cov_kwds={"groups": d.index.factorize()[0]})
            rows.append({"signal": sig, "window": w, "period": per, "slope bp per bp": f_.params.iloc[1], "t": f_.tvalues.iloc[1]})
print(pd.DataFrame(rows).round(2).pivot_table(index=["signal", "window"], columns="period", values=["slope bp per bp", "t"]).round(2).to_string())
""")
md(r"""
## Tradable version
**Rule.** At settlement $T$, if `fr_prev` is above the coin's in-sample q-th percentile, go long for $h$ minutes;
equal-weight across coins triggering at the same $T$; net 4bp.
""")
code(r"""
out = []
for q in [0.8, 0.9, 0.95]:
    thr = D[D["is"]].groupby("sym").fr_prev.quantile(q); sig = D.fr_prev > D.sym.map(thr)
    for w in ["post30" if "post30" in D else "post10", "post60", "post120"]:
        for per, m in [("in-sample", D["is"]), ("out-of-sample", ~D["is"])]:
            t = D[m & sig]; g = t[w].groupby(t.index).mean() - 4
            out.append({"q": q, "hold": w, "period": per, "events": len(g), "bp/trade": g.mean(), "t": g.mean() / g.std() * np.sqrt(len(g)) if len(g) > 2 else np.nan})
print(pd.DataFrame(out).round(2).to_string(index=False))
""")
md(r"""
**Interpretation.**
- **No pre-settlement selling.** The pre-settlement windows show no reliable negative slope on known funding
  (in-sample t between −1.3 and +0.7; out of sample only the last 10 minutes reach t ≈ −2.0, a single window among many).
- **A post-settlement rebound after extreme funding, in-sample only.** Above the in-sample 95th percentile, longs
  from $T$ earn +23 to +33bp net (t ≈ 3.3–3.6) in 2020–23, mostly in the 2021 mania. Out of sample, funding rarely
  gets that high (28 events), and the result is insignificant (t ≈ 0.7). At looser thresholds it fails (t < 1.4).
- The intended secondary signal (the rate applied at $T$) correlates with the *placebo* window (t ≈ 2.6–3.2), i.e. it
  partly records price moves inside the funding interval. That is why it cannot be used.

**Verdict: not tradable.** The effect lives in the 2021 extreme-funding episodes; out of sample there are too few
such events and nothing at normal funding levels clears costs.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
