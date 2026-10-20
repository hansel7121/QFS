"""Generate eda.ipynb: bitcoin around the 15:00-16:00 ET fixing window used by US spot ETFs (CF Benchmarks BRR NY)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# BTCFix Pitch EDA: do spot-ETF creations push bitcoin in the 3–4pm ET fixing window?

**The mechanism.** US spot bitcoin ETFs (launched 2024-01-11) strike NAV on the CF Benchmarks New York reference
rate, computed from trades between 15:00 and 16:00 ET. Cash creations and redemptions are executed against that
window, so on days with flows the issuer or its market makers trade bitcoin in that hour. Flows chase performance, so
a strong US day should bring buying in the window (continuation), then reversal after it. Daily flow data is not
freely downloadable (Farside / issuer files block scripts), so the test uses the day's return as the flow proxy.

**Design.** Binance BTCUSDT perpetual 5-minute bars (`Funding Pitch/data`), New York time. Day = 09:30→15:00 ET;
fix = 15:00→16:00; after = 16:00→18:00. US trading weekdays before and after the ETF launch, plus weekends (no ETF
flow) as a control. Placebo: other one-hour windows.
""")
code(r"""
import pandas as pd, numpy as np, statsmodels.api as sm, holidays
k=pd.read_csv("../Funding Pitch/data/klines_BTCUSDT.csv",parse_dates=["open_time"])
k["t"]=k.open_time.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
k=k.set_index("t")
c=np.log(k.close); v=k.quote_volume
k["d"]=k.index.normalize(); k["hm"]=k.index.strftime("%H:%M")
C=k.pivot_table(index="d",columns="hm",values="close"); Vq=k.pivot_table(index="d",columns="hm",values="quote_volume")
us=holidays.NYSE(years=range(2020,2027))
C.index=C.index.tz_localize(None); Vq.index=Vq.index.tz_localize(None)
wd=(C.index.dayofweek<5)&np.array([d not in us for d in C.index])
L=np.log(C)
def r(a,b): return (L[b]-L[a])*1e4      # close of bar starting a -> close of bar starting b
# windows: 09:30->14:55 (US day, signal), 14:55->15:55 (fixing window 15:00-16:00), 15:55->17:55 (after)
D=pd.DataFrame({"day":r("09:25","14:55"),"fix":r("14:55","15:55"),"after":r("15:55","17:55"),
                "vol_fix":Vq.loc[:,"15:00":"15:55"].sum(1)/(Vq.loc[:,"10:00":"14:55"].sum(1)/5)},index=C.index)   # fix-hour volume / average hour 10:00-15:00
D["wd"]=wd; D["post"]=D.index>="2024-01-11"
for grp,m in [("weekday pre-ETF",D.wd&~D.post),("weekday post-ETF",D.wd&D.post),("weekend post-ETF",~D.wd&D.post)]:
    d=D[m].dropna()
    f=sm.OLS(d.fix,sm.add_constant(d.day)).fit(cov_type="HC1"); g=sm.OLS(d.after,sm.add_constant(d.day)).fit(cov_type="HC1")
    print(f"{grp:18s} n {len(d)} | fix-hour vol / avg hour {d.vol_fix.median():.2f} | fix ~ day: b {f.params.day:+.3f} (t {f.tvalues.day:+.2f}) | after ~ day: b {g.params.day:+.3f} (t {g.tvalues.day:+.2f}) | mean fix {d.fix.mean():+.1f}")
# placebo hours post-ETF weekdays: other 1h windows
d=D[D.wd&D.post]
for a,b in [("11:55","12:55"),("12:55","13:55"),("13:55","14:55"),("16:55","17:55")]:
    y_=r(a,b)[D.wd&D.post]; f=sm.OLS(y_,sm.add_constant(r("09:25",a)[D.wd&D.post])).fit(cov_type="HC1")
    print(f"placebo {a}->{b}: b {f.params.iloc[1]:+.3f} (t {f.tvalues.iloc[1]:+.2f})")

""")
md(r"""
**Interpretation.**
- **No fixing-window effect.** The US-day return does not predict the 15:00–16:00 ET return on weekdays, either before
  (t −0.1) or after (t −0.7) the ETF launch, and not on weekends. Placebo hours look the same.
- The 16:00–18:00 reversal coefficient is mildly negative on post-ETF weekdays (t −1.9), but the 16:55–17:55 placebo
  has the same sign and size. It is generic short-horizon reversal.

**Verdict: dead.** If ETF flows trade in the window, liquidity absorbs them.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
