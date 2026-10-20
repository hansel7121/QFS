"""Is the own-discount z-score signal anything more than short-term price reversal? (training period 2012-2019 only)

Each week, across BDCs, regress the z-score on recent return vs peers and keep the residual: the part of "cheap vs own
history" that recent price moves do not explain. Then rerun the weekly rank-correlation test (same as Fig 7) on the residual.
Outputs: report_figs/f7c_residual.png, report_figs/residual_numbers.json.
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, FIG = ROOT / "data", ROOT / "report_figs"
SPLIT = pd.Timestamp("2020-01-01")
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
BLUE, ORANGE, GREY = "#2f5d8a", "#d17a2b", "#888888"


def hac_t(x, lags):
    x = pd.Series(x).dropna()
    return sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).tvalues.iloc[0]


# ---------------------------------------------------------------- data (identical to build_report_figs.py)
H = pd.read_csv(DATA / "nav_history.csv", parse_dates=["period", "public"])
H = H[H.clean & (H.ticker != "GECC") & ~((H.ticker == "CSWC") & (H.period < "2016-01-01"))].sort_values(["ticker", "period"])
p = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
C = p.pivot(index="date", columns="ticker", values="close")
A = p.pivot(index="date", columns="ticker", values="adj_close")
dates = C.index
bdcs = [t for t in H.ticker.unique() if t in C]
R = np.log(A[bdcs]).diff()
rel = R.sub(R.mean(1), axis=0)

N = pd.DataFrame(index=dates, columns=bdcs, dtype=float)
for t, g in H[H.ticker.isin(bdcs)].groupby("ticker"):
    eff = dates[np.minimum(np.searchsorted(dates, g.public.dt.normalize(), side="right"), len(dates) - 1)]
    s = pd.Series(g.nav_ps.values, index=eff); s = s[~s.index.duplicated(keep="last")]
    N[t] = s.reindex(dates).ffill()
    last = pd.Series(s.index, index=s.index).reindex(dates).ffill()
    N.loc[(dates.to_series() - last).dt.days > 150, t] = np.nan
disc = np.log(C[bdcs] / N)
last_tr = int(np.searchsorted(dates, SPLIT)) - 1

mu = disc.rolling(252, min_periods=180).mean().shift(1); sd = disc.rolling(252, min_periods=180).std().shift(1)
Z = (disc - mu) / sd
past = {k: rel.rolling(k).sum() for k in (5, 20, 60)}


def fwd_ret(H_):
    return sum(rel.shift(-k) for k in range(2, H_ + 2))


def train_weeks(H_):
    ix = dates[::5]
    return ix[np.searchsorted(dates, ix) + H_ + 1 <= last_tr]


def residual(d, ctrls):
    """Cross-sectional residual of Z on the control returns at date d."""
    df = pd.DataFrame({"z": Z.loc[d], **{f"p{k}": past[k].loc[d] for k in ctrls}}).dropna()
    if len(df) < 10: return pd.Series(dtype=float)
    return sm.OLS(df.z, sm.add_constant(df.drop(columns="z"))).fit().resid


def weekly_ic(sigfun, H_):
    fwd = fwd_ret(H_); out = {}
    for d in train_weeks(H_):
        s = sigfun(d); y = fwd.loc[d].reindex(s.index)
        ok = s.notna() & y.notna()
        if ok.sum() >= 10: out[d] = s[ok].corr(y[ok], method="spearman")
    return pd.Series(out)


signals = {
    "z-score (as in writeup)": lambda d: Z.loc[d],
    "z-score, past 20d return removed": lambda d: residual(d, [20]),
    "z-score, past 5/20/60d returns removed": lambda d: residual(d, [5, 20, 60]),
    "past 20d return vs peers": lambda d: past[20].loc[d],
}
rows = []
for H_ in (5, 20, 60):
    for lab, f in signals.items():
        rho = weekly_ic(f, H_)
        rows.append({"H": H_, "signal": lab, "ic": float(rho.mean()), "t": float(hac_t(rho, max(1, H_ // 5))), "weeks": len(rho)})
T = pd.DataFrame(rows)

# how much of the z-score is just recent price moves? weekly cross-sectional R^2 of Z on past 20d return
r2 = {}
for d in train_weeks(20):
    df = pd.DataFrame({"z": Z.loc[d], "p": past[20].loc[d]}).dropna()
    if len(df) >= 10: r2[d] = df.z.corr(df.p) ** 2
r2 = pd.Series(r2)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].hist(r2, bins=25, color=BLUE)
ax[0].axvline(r2.median(), c=ORANGE, ls="--", label=f"median {r2.median():.2f}")
ax[0].set_xlabel("share of z-score explained by past 20d return\n(weekly cross-sectional R²)"); ax[0].set_ylabel("number of weeks")
ax[0].set_title("(a) How much of the z-score is recent price moves?"); ax[0].legend(fontsize=7)

labs = list(signals); cols = [BLUE, "#6f9bc4", "#b3cde3", GREY]; w = 0.2; k = np.arange(3)
for i, (lab, c) in enumerate(zip(labs, cols)):
    z = T[T.signal == lab]
    ax[1].bar(k + (i - 1.5) * w, z.t, w, color=c, label=lab)
ax[1].axhline(-2, c="r", ls="--", lw=0.8); ax[1].axhline(0, c="k", lw=0.8)
ax[1].set_xticks(k); ax[1].set_xticklabels(["next 5 days", "next 20 days", "next 60 days"])
ax[1].set_ylabel("t-stat of weekly rank correlation\nwith future return vs peers")
ax[1].set_title("(b) Signal strength before and after removing reversal (2012–2019)"); ax[1].legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.savefig(FIG / "f7c_residual.png", bbox_inches="tight"); plt.close()

json.dump({"table": rows, "r2_median": float(r2.median()), "r2_mean": float(r2.mean())},
          open(FIG / "residual_numbers.json", "w"), indent=1)
print(T.pivot(index="signal", columns="H", values="t").round(2).loc[labs])
print(T.pivot(index="signal", columns="H", values="ic").round(3).loc[labs])
print("median weekly R^2 of z on past 20d:", round(r2.median(), 3))
