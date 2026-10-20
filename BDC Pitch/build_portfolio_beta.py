"""Beta of the whole long/short portfolio to the BDC sector through the training period (2013-2019).

  ex-ante  : each day, sum over held positions of (dollar weight x that BDC's trailing-252d beta), longs minus shorts,
             i.e. the book's beta known at the time
  realized : rolling 126-day regression of the strategy's daily return on the sector's daily return
Shown for the neutral version (relative z-score, beta-neutral).
Output: report_figs/n4_portfolio_beta_is.png
"""

import runpy, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path(__file__).parent
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(ROOT / "build_neutral.py"))
Z, Zrel, R, sector, BETA, DV, dates, bdcs = (G[k] for k in ("Z", "Zrel", "R", "sector", "BETA", "DV", "dates", "bdcs"))
bH, bq, STEP, save, SPLIT, strat = G["bH"], G["bq"], G["STEP"], G["save"], G["SPLIT"], G["strat"]
BLUE, GREEN, GREY = G["BLUE"], G["GREEN"], G["GREY"]


def book_beta(sig, weighting):
    """Daily ex-ante beta of the book: same position rules as strat() in build_neutral.py."""
    wl = np.zeros((len(dates), len(bdcs))); ws = np.zeros_like(wl); col = {t: i for i, t in enumerate(bdcs)}
    for i in range(4, len(dates) - 2, STEP):
        s = sig.iloc[i].where(DV.iloc[i].notna() & BETA.iloc[i].notna()).dropna()
        if len(s) < 9: continue
        lo = s[s <= s.quantile(bq)].index; hi_ = s[s >= s.quantile(1 - bq)].index
        bl, bs = BETA.iloc[i][lo].mean(), BETA.iloc[i][hi_].mean()
        gl, gs = (2 * bs / (bl + bs), 2 * bl / (bl + bs)) if weighting == "beta-neutral" else (1.0, 1.0)
        k = 1 / (bH / STEP)
        wl[i + 2:i + 2 + bH, [col[x] for x in lo]] += gl / len(lo) * k
        ws[i + 2:i + 2 + bH, [col[x] for x in hi_]] += gs / len(hi_) * k
    # beta frozen at each position's formation date would need per-tranche bookkeeping; today's trailing beta is used,
    # which is still known on that day (BETA is lagged one day)
    WL, WS = pd.DataFrame(wl, index=dates, columns=bdcs), pd.DataFrame(ws, index=dates, columns=bdcs)
    return (WL * BETA).sum(1) - (WS * BETA).sum(1), WL.sum(1) - WS.sum(1)


IS = (dates >= "2013-03-01") & (dates < SPLIT)
VERS = [("relative z-score, beta-neutral", Zrel, "beta-neutral", GREEN)]
fig, ax = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
for lab, sig, w, c in VERS:
    bb, net_dollar = book_beta(sig, w)
    net = strat(sig, bH, bq, STEP, w)[0]
    net = net[net.index < SPLIT]
    sec = sector.reindex(net.index)
    rb = net.rolling(126, min_periods=90).cov(sec) / sec.rolling(126, min_periods=90).var()
    full = sm.OLS(net, sm.add_constant(sec)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    bb_is = bb[IS]
    ax[0].plot(bb_is.index, bb_is, color=c, lw=1, label=f"{lab} (avg {bb_is.mean():+.3f})")
    ax[1].plot(rb.index, rb, color=c, lw=1.2, label=f"{lab} (whole period {full.params.iloc[1]:+.3f}, t = {full.tvalues.iloc[1]:.1f})")
    print(lab, "ex-ante avg", round(bb_is.mean(), 3), "range", round(bb_is.min(), 3), round(bb_is.max(), 3),
          "| realized IS", round(full.params.iloc[1], 3), "t", round(full.tvalues.iloc[1], 2), "| net $ avg", round(net_dollar[IS].mean(), 3))
for a in ax:
    a.axhline(0, c="k", lw=0.8); a.legend(fontsize=7.5, loc="upper left")
ax[0].set_ylabel("ex-ante book beta\n(long β × weight − short β × weight)")
ax[0].set_title("(a) Portfolio beta known at the time: positions held each day × each BDC's trailing-252-day beta")
ax[1].set_ylabel("realized beta\n(rolling 126 days)")
ax[1].set_title("(b) Portfolio beta measured from the strategy's actual daily returns vs the BDC sector")
fig.suptitle("Beta of the whole long/short portfolio to the BDC sector, training period 2013–2019", fontsize=10)
save("n4_portfolio_beta_is")
