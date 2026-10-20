"""How the long- and short-basket betas are computed (write-up version: z-score, equal $, hold 12d, 25% buckets).

  (a) one BDC, one formation date: its daily returns vs the equal-weight BDC sector over the 252 trading days BEFORE
      the date; the OLS slope is its beta (cov / var), the value used for that trade
  (b) that BDC's beta over time (same 252-day window, rolled forward one day at a time, lagged one day)
  (c) the same formation date: every BDC's beta, coloured by basket; the basket beta is the simple average
      (equal $ within a basket)
  (d) basket betas over time: ex-ante (average of member betas, as in (c), weighted by the overlapping tranches held each
      day) vs realized (rolling 252-day beta of each basket's actual daily return to the sector)
Output: report_figs/n3_beta_method.png
"""

import runpy, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(ROOT / "build_neutral.py"))
Z, R, sector, BETA, DV, dates, bdcs = G["Z"], G["R"], G["sector"], G["BETA"], G["DV"], G["dates"], G["bdcs"]
bH, bq, STEP, save = G["bH"], G["bq"], G["STEP"], G["save"]
BLUE, ORANGE, GREY = G["BLUE"], G["ORANGE"], G["GREY"]
rr = R.fillna(0)

# rebuild the write-up version's daily weights (same loop as strat(), equal $)
wl = np.zeros((len(dates), len(bdcs))); ws = np.zeros_like(wl); col = {t: i for i, t in enumerate(bdcs)}
baskets = {}
for i in range(4, len(dates) - 2, STEP):
    s = Z.iloc[i].where(DV.iloc[i].notna() & BETA.iloc[i].notna()).dropna()
    if len(s) < 9: continue
    lo = s[s <= s.quantile(bq)].index; hi_ = s[s >= s.quantile(1 - bq)].index
    baskets[dates[i]] = (list(lo), list(hi_), s.index.tolist())
    k = 1 / (bH / STEP)
    wl[i + 2:i + 2 + bH, [col[x] for x in lo]] += k / len(lo); ws[i + 2:i + 2 + bH, [col[x] for x in hi_]] += k / len(hi_)
WL, WS = pd.DataFrame(wl, index=dates, columns=bdcs), pd.DataFrame(ws, index=dates, columns=bdcs)
on = (WL.sum(1) > 0) & (dates >= "2013-03-01")
exL = ((WL * BETA).sum(1) / WL.sum(1))[on]; exS = ((WS * BETA).sum(1) / WS.sum(1))[on]
legL = ((WL * rr).sum(1) / WL.sum(1))[on]; legS = ((WS * rr).sum(1) / WS.sum(1))[on]
sec = sector.reindex(legL.index)
reL = legL.rolling(252, min_periods=126).cov(sec) / sec.rolling(252, min_periods=126).var()
reS = legS.rolling(252, min_periods=126).cov(sec) / sec.rolling(252, min_periods=126).var()

# example formation date and BDC
d0 = min(baskets, key=lambda d: abs(d - pd.Timestamp("2019-03-01")))
lo0, hi0, univ0 = baskets[d0]
ex = max(lo0, key=lambda t: BETA.loc[d0, t])                       # highest-beta name in the long basket that day
i0 = dates.get_loc(d0)
win = slice(i0 - 252, i0)                                          # the 252 days before formation (beta is lagged one day)
x, y = sector.iloc[win], R[ex].iloc[win]
ok = x.notna() & y.notna()
b_ex = np.cov(y[ok], x[ok])[0, 1] / x[ok].var()

fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
a = ax[0, 0]
a.scatter(x[ok] * 100, y[ok] * 100, s=6, color=BLUE, alpha=0.5)
xx = np.linspace(x[ok].min(), x[ok].max(), 10)
a.plot(xx * 100, (y[ok].mean() + b_ex * (xx - x[ok].mean())) * 100, color=ORANGE, lw=2, label=f"slope = beta = {b_ex:.2f}")
a.axhline(0, c="k", lw=0.6); a.axvline(0, c="k", lw=0.6)
a.set_xlabel("BDC sector return that day (%, equal-weight average of all BDCs)"); a.set_ylabel(f"{ex} return that day (%)")
a.set_title(f"(a) Step 1: one BDC's beta. {ex}, 252 trading days before {d0:%d %b %Y}\nbeta = cov({ex}, sector) / var(sector)")
a.legend(fontsize=8)

a = ax[0, 1]
bser = BETA[ex].dropna()
a.plot(bser.index, bser, color=BLUE)
a.scatter([d0], [BETA.loc[d0, ex]], color=ORANGE, zorder=5, label=f"value used on {d0:%d %b %Y}: {BETA.loc[d0, ex]:.2f}")
a.axhline(1, c="k", lw=0.6, ls=":"); a.set_ylabel("beta to BDC sector")
a.set_title(f"(b) Step 2: re-estimated every day on the trailing 252 days\n(only past data; clipped to 0.2–2.5)")
a.legend(fontsize=8)

a = ax[1, 0]
b0 = BETA.loc[d0, univ0].sort_values()
cols = [BLUE if t in lo0 else ORANGE if t in hi0 else "#cccccc" for t in b0.index]
a.bar(range(len(b0)), b0.values, color=cols)
a.set_xticks(range(len(b0))); a.set_xticklabels(b0.index, rotation=90, fontsize=6)
bl, bs = b0[lo0].mean(), b0[hi0].mean()
a.axhline(bl, color=BLUE, ls="--", lw=1.2, label=f"long basket (cheapest {bq:.0%}) average = {bl:.2f}")
a.axhline(bs, color=ORANGE, ls="--", lw=1.2, label=f"short basket (richest {bq:.0%}) average = {bs:.2f}")
a.set_ylabel("beta to BDC sector"); a.legend(fontsize=7, loc="upper left")
a.set_title(f"(c) Step 3: basket beta = average of member betas, {d0:%d %b %Y}\nblue = long, orange = short, grey = not traded")

a = ax[1, 1]
m = lambda s: s.resample("M").mean()
a.plot(m(exL).index, m(exL), color=BLUE, lw=1.2, label="long basket, ex-ante (avg of member betas)")
a.plot(m(exS).index, m(exS), color=ORANGE, lw=1.2, label="short basket, ex-ante")
a.plot(reL.index, reL, color=BLUE, lw=1, ls=":", label="long basket, realized (rolling 252d beta of its returns)")
a.plot(reS.index, reS, color=ORANGE, lw=1, ls=":", label="short basket, realized")
a.axvline(pd.Timestamp("2020-01-01"), c="r", ls="--", lw=1); a.axhline(1, c="k", lw=0.6)
a.set_ylabel("beta to BDC sector"); a.legend(fontsize=6.5, loc="lower left")
a.set_title(f"(d) Step 4: each day, average over the {bH // STEP} overlapping tranches held\n"
            f"avg ex-ante: long {exL.mean():.2f}, short {exS.mean():.2f}; realized: long {reL.mean():.2f}, short {reS.mean():.2f}")
save("n3_beta_method")
print(d0.date(), ex, round(b_ex, 3), round(BETA.loc[d0, ex], 3), "basket", round(bl, 3), round(bs, 3))
print("ex-ante", round(exL.mean(), 3), round(exS.mean(), 3), "realized", round(reL.mean(), 3), round(reS.mean(), 3))
