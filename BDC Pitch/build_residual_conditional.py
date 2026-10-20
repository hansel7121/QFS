"""Where does the reversal-free part of the z-score still predict returns? (training period 2012-2019 only)

Residual = weekly cross-sectional residual of the own-discount z-score on past 5/20/60-day return vs peers
(see build_residual_test.py). If the discount mechanism is real, the residual should work
  (1) when the NAV is fresh (just published) rather than stale, and
  (2) when the BDC trades at a premium (it can issue shares) or a deep discount (it can buy back), not in the middle.
Method: Fama-MacBeth. Each week, regress the percentile rank of the next-H-day return vs peers on the standardized
residual, interacted with group dummies (plus group intercepts). The weekly slope for each group is averaged over
weeks; t-stat is HAC. A negative slope = cheap-vs-own-history (after removing reversal) outperforms.
Outputs: report_figs/f7d_residual_conditional.png, report_figs/residual_conditional_numbers.json.
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
AGE = pd.DataFrame(index=dates, columns=bdcs, dtype=float)       # trading days since the current NAV took effect
pos = pd.Series(np.arange(len(dates)), index=dates)
for t, g in H[H.ticker.isin(bdcs)].groupby("ticker"):
    eff = dates[np.minimum(np.searchsorted(dates, g.public.dt.normalize(), side="right"), len(dates) - 1)]
    s = pd.Series(g.nav_ps.values, index=eff); s = s[~s.index.duplicated(keep="last")]
    N[t] = s.reindex(dates).ffill()
    last = pd.Series(s.index, index=s.index).reindex(dates).ffill()
    N.loc[(dates.to_series() - last).dt.days > 150, t] = np.nan
    AGE[t] = pos.values - pos.reindex(last.values).values
disc = np.log(C[bdcs] / N)
LVL = np.exp(disc) - 1
last_tr = int(np.searchsorted(dates, SPLIT)) - 1

mu = disc.rolling(252, min_periods=180).mean().shift(1); sd = disc.rolling(252, min_periods=180).std().shift(1)
Z = (disc - mu) / sd
past = {k: rel.rolling(k).sum() for k in (5, 20, 60)}


def fwd_ret(H_):
    return sum(rel.shift(-k) for k in range(2, H_ + 2))


def train_weeks(H_):
    ix = dates[::5]
    return ix[np.searchsorted(dates, ix) + H_ + 1 <= last_tr]


def residual(d):
    df = pd.DataFrame({"z": Z.loc[d], **{f"p{k}": past[k].loc[d] for k in past}}).dropna()
    if len(df) < 10: return pd.Series(dtype=float)
    r = sm.OLS(df.z, sm.add_constant(df.drop(columns="z"))).fit().resid
    return (r - r.mean()) / r.std()


# ---------------------------------------------------------------- groupings (all known at the signal date)
FRESH_DAYS = 15
groupings = {
    "NAV age": lambda d: pd.Series(np.where(AGE.loc[d] <= FRESH_DAYS, f"fresh (≤{FRESH_DAYS}d since NAV)", f"stale (>{FRESH_DAYS}d)"),
                                   index=bdcs).where(AGE.loc[d].notna()),
    "discount level": lambda d: pd.Series(np.select([LVL.loc[d] >= 0, LVL.loc[d] <= -0.15], ["premium (price ≥ NAV)", "deep discount (≤ −15%)"],
                                                    "middle (−15% to 0)"), index=bdcs).where(LVL.loc[d].notna()),
}
ORDER = {"NAV age": [f"fresh (≤{FRESH_DAYS}d since NAV)", f"stale (>{FRESH_DAYS}d)"],
         "discount level": ["premium (price ≥ NAV)", "middle (−15% to 0)", "deep discount (≤ −15%)"]}


def fm_slopes(gname, H_):
    fwd = fwd_ret(H_); gf = groupings[gname]; out, cnt = [], []
    for d in train_weeks(H_):
        r = residual(d)
        df = pd.DataFrame({"r": r, "y": fwd.loc[d].reindex(r.index), "g": gf(d).reindex(r.index)}).dropna()
        if len(df) < 10: continue
        df["y"] = df.y.rank(pct=True) - 0.5
        X = pd.DataFrame(index=df.index)
        row = {}
        for g in ORDER[gname]:
            m = (df.g == g).astype(float)
            if m.sum() < 3: continue                                 # too few names in this group this week
            X[f"a:{g}"] = m; X[f"b:{g}"] = m * df.r
        if X.shape[1] == 0: continue
        b = sm.OLS(df.y, X).fit().params
        out.append({g: b.get(f"b:{g}", np.nan) for g in ORDER[gname]})
        cnt.append(df.g.value_counts())
    out = pd.DataFrame(out); cnt = pd.DataFrame(cnt).fillna(0)
    return [{"grouping": gname, "group": g, "H": H_, "slope": float(out[g].mean()), "t": float(hac_t(out[g], max(1, H_ // 5))),
             "weeks": int(out[g].notna().sum()), "avg_names": float(cnt[g].mean()) if g in cnt else 0.0} for g in ORDER[gname]]


rows = []
for H_ in (5, 20, 60):
    # unconditional residual slope, for reference
    fwd = fwd_ret(H_); u = []
    for d in train_weeks(H_):
        r = residual(d); y = fwd.loc[d].reindex(r.index); ok = y.notna()
        if ok.sum() >= 10:
            u.append(sm.OLS(y[ok].rank(pct=True) - 0.5, sm.add_constant(r[ok])).fit().params.iloc[1])
    rows.append({"grouping": "all", "group": "all BDCs", "H": H_, "slope": float(np.mean(u)), "t": float(hac_t(u, max(1, H_ // 5))),
                 "weeks": len(u), "avg_names": np.nan})
    for gname in groupings:
        rows += fm_slopes(gname, H_)
T = pd.DataFrame(rows)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, gridspec_kw={"width_ratios": [1, 2, 3]})
for a, gname, groups in zip(ax, ["all", "NAV age", "discount level"], [["all BDCs"], ORDER["NAV age"], ORDER["discount level"]]):
    k = np.arange(len(groups)); w = 0.25
    for i, (H_, c) in enumerate(zip((5, 20, 60), ["#b3cde3", "#6f9bc4", BLUE])):
        z = T[(T.grouping == gname) & (T.H == H_)].set_index("group").reindex(groups)
        a.bar(k + (i - 1) * w, z.t, w, color=c, label=f"next {H_} days")
    a.axhline(-2, c="r", ls="--", lw=0.8); a.axhline(2, c="r", ls="--", lw=0.8); a.axhline(0, c="k", lw=0.8)
    a.set_xticks(k); a.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=7)
    a.set_title({"all": "(a) all BDCs", "NAV age": "(b) by how fresh the NAV is", "discount level": "(c) by discount level"}[gname])
ax[0].set_ylabel("t-stat: reversal-free z-score\nvs future return vs peers"); ax[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Does the discount signal survive after removing reversal, in the places the mechanism predicts? (2012–2019)", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "f7d_residual_conditional.png", bbox_inches="tight"); plt.close()

json.dump({"fresh_days": FRESH_DAYS, "table": rows}, open(FIG / "residual_conditional_numbers.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(T.pivot_table(index=["grouping", "group"], columns="H", values="t", sort=False).round(2))
print(T[T.H == 20][["group", "avg_names", "weeks"]].round(1).to_string(index=False))
st = LVL[dates < SPLIT].stack()
print("in-sample share premium:", round((st >= 0).mean(), 3), " deep (<=-15%):", round((st <= -0.15).mean(), 3))
