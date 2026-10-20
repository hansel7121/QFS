"""Generate eda.ipynb: bitcoin around crypto derivative expiries: CME futures (last Friday, 15:00-16:00 London) and Deribit options (Fridays 08:00 UTC)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# CMEExpiry Pitch EDA: do crypto derivative expiries move spot?

**Mechanisms.**
* **CME bitcoin futures** settle on the last Friday of the month to the CF Benchmarks reference rate, computed from
  15:00–16:00 London trades. Cash-and-carry funds (long spot or ETF, short CME), much bigger since spot ETFs launched in
  2024, roll or unwind against that window.
* **Deribit options** (about 80% of crypto option open interest) expire on Fridays at 08:00 UTC against a 30-minute
  index average (07:30–08:00). The monthly (last-Friday) expiry carries most of the open interest. Dealers unwind delta
  hedges into the settlement.

**Design.** Binance perpetual 5-minute bars (`Funding Pitch/data`). Expiry Fridays against other Fridays (CME) or
Thursdays / non-Fridays (Deribit). 2020–23 vs 2024–.
""")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
k=pd.read_csv("../Funding Pitch/data/klines_BTCUSDT.csv",parse_dates=["open_time"])
k["t"]=k.open_time.dt.tz_localize("UTC").dt.tz_convert("Europe/London")
k["d"]=k.t.dt.tz_localize(None).dt.normalize(); k["hm"]=k.t.dt.strftime("%H:%M")
L=np.log(k.pivot_table(index="d",columns="hm",values="close"))
def r(a,b): return (L[b]-L[a])*1e4
D=pd.DataFrame({"pre_window":r("12:55","14:55"),"window":r("14:55","15:55"),"after":r("15:55","18:55"),"day_before":(L["14:55"]-L["14:55"].shift(1))*1e4},index=L.index)
fri=D[D.index.dayofweek==4].copy()
fri["last"]=fri.groupby(fri.index.to_period("M")).cumcount(ascending=False)==0
fri["quarterly"]=fri["last"]&fri.index.month.isin([3,6,9,12])
for per,m in [("2020-2023",fri.index<"2024"),("2024- (spot ETFs, bigger basis trade)",fri.index>="2024")]:
    for c in ["pre_window","window","after"]:
        a=fri[m&fri["last"]][c].dropna(); b=fri[m&~fri["last"]][c].dropna(); t,_=stats.ttest_ind(a,b,equal_var=False)
        q=fri[m&fri["quarterly"]][c].dropna()
        print(f"{per:38s} {c:10s}: expiry Fri {a.mean():+.1f} (n {len(a)}) vs other Fri {b.mean():+.1f}  t {t:+.2f} | quarterly {q.mean():+.1f} (n {len(q)}); |abs| expiry {a.abs().mean():.0f} vs {b.abs().mean():.0f}")
print("\n==== Deribit options expiry: Fridays 08:00 UTC (monthly = last Friday)")
k["tu"]=k.open_time; k["du"]=k.tu.dt.normalize(); k["hmu"]=k.tu.dt.strftime("%H:%M")
LU=np.log(k.pivot_table(index="du",columns="hmu",values="close"))
ru=lambda a,b:(LU[b]-LU[a])*1e4
U=pd.DataFrame({"into":ru("05:55","07:55"),"after":ru("07:55","09:55"),"absinto":ru("05:55","07:55").abs()},index=LU.index)
U["dow"]=U.index.dayofweek
f=U[U.dow==4].copy(); f["last"]=f.groupby(f.index.to_period("M")).cumcount(ascending=False)==0
th=U[U.dow==3]
for per,m in [("2020-2023",lambda x:x.index<"2024"),("2024-",lambda x:x.index>="2024")]:
    a=f[m(f)&f["last"]]; b=f[m(f)&~f["last"]]; c=th[m(th)]
    for col in ["into","after","absinto"]:
        t,_=stats.ttest_ind(a[col].dropna(),c[col].dropna(),equal_var=False)
        print(f"{per} {col:8s}: monthly-expiry Fri {a[col].mean():+.1f} | weekly Fri {b[col].mean():+.1f} | Thu {c[col].mean():+.1f}  (t monthly vs Thu {t:+.2f})")

""")
md(r"""
**Interpretation (CME).** Nothing consistent. Windows around the 15:00–16:00 London settlement flip sign between
2020–23 and 2024– (all |t| < 2).

**Interpretation (Deribit).** The 2 hours into the monthly expiry (06:00 → 08:00 UTC) show a drop since 2024:
−32bp vs +3bp on Thursdays (t −3.7, n 33). Weekly Fridays show nothing (+1bp). In 2020–23 the monthly drop is −8bp
(t −0.5).
""")
md("## Deribit monthly-expiry window across coins and years")
code(r"""
import pandas as pd, numpy as np
from scipy import stats
for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
    k = pd.read_csv(f"../Funding Pitch/data/klines_{s}.csv", parse_dates=["open_time"])
    k["du"] = k.open_time.dt.normalize(); k["hmu"] = k.open_time.dt.strftime("%H:%M")
    L = np.log(k.pivot_table(index="du", columns="hmu", values="close"))
    df = pd.DataFrame({"into": (L["07:55"] - L["05:55"]) * 1e4}); df["dow"] = df.index.dayofweek
    f = df[df.dow == 4].copy(); f["last"] = f.groupby(f.index.to_period("M")).cumcount(ascending=False) == 0
    m = f[f["last"]]; a = m[m.index >= "2024"].into; oth = df[(df.dow != 4) & (df.index >= "2024")].into
    print(s, "2024– monthly-expiry Fri", f"{a.mean():+.1f}bp (n {len(a)}, share negative {(a < 0).mean():.0%})", "vs non-Fridays", f"{oth.mean():+.1f}",
          "t", f"{stats.ttest_ind(a, oth, equal_var=False)[0]:+.2f}", "| by year:", m.groupby(m.index.year).into.mean().round(0).to_dict())
""")
md(r"""
**Interpretation.**
- **Broad across coins since 2024:** BTC −32bp (85% of expiries negative), ETH −42bp (t −3.4), XRP −28bp (t −2.0),
  SOL −32bp (t −1.5).
- **But not a clean out-of-sample result.** The direction was found in this analysis, not predicted. 2020–23 is
  the same sign but small and insignificant, and 2026 to date is weaker (BTC −9bp).
- **Plausible mechanism:** in a call-heavy market, dealers short calls are long delta hedges and sell them as the
  calls expire into the 07:30–08:00 settlement average. Testing it needs historical Deribit open interest by strike
  (not free).

**Verdict: a lead, not a pass.** Re-test on events after October 2026, and with Deribit open-interest data if it
can be had, before treating it as tradable.
""")
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nbf.write(nb, "eda.ipynb")
print("wrote eda.ipynb")
