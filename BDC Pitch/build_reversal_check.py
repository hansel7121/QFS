"""Is the own-discount z-score just short-term reversal in disguise? (training period 2012-2019 only)

Three checks:
  (A) Premise: between NAV releases the denominator is frozen, so how much of the day-to-day movement in z comes from the
      price, from NAV updates, and from the drift of the trailing mean/std?
  (B) Double sort: each week, split BDCs into 3 groups by past k-day return vs peers; within each group, split by z into
      halves; spread = (cheap-half minus rich-half) next-H-day return vs peers. If the spread is positive in every
      past-return group, z adds something beyond reversal. The reverse sort (control z, sort by past return) is shown too.
  (C) Regression (Fama-MacBeth): r(t->t+h) = a + b z + c r(t-k->t) + e, each week across BDCs, both regressors
      standardized; average b and c over weeks, HAC t-stats.
Outputs: report_figs/f7e_reversal_check.png, report_figs/reversal_check_numbers.json.
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
TR = dates < SPLIT

mu = disc.rolling(252, min_periods=180).mean().shift(1); sd = disc.rolling(252, min_periods=180).std().shift(1)
Z = (disc - mu) / sd
past = {k: rel.rolling(k).sum() for k in (5, 20, 60)}


def fwd_ret(H_):
    return sum(rel.shift(-k) for k in range(2, H_ + 2))


def train_weeks(H_):
    ix = dates[::5]
    return ix[np.searchsorted(dates, ix) + H_ + 1 <= last_tr]


OUT = {}

# ---------------------------------------------------------------- (A) where do changes in z come from?
# disc_t - disc_{t-1} = (log C_t - log C_{t-1}) - (log N_t - log N_{t-1}); z also moves when mu/sd roll.
dP = np.log(C[bdcs]).diff().where(disc.notna() & disc.shift(1).notna())
dN = -np.log(N).diff().where(disc.notna() & disc.shift(1).notna())
dZ = Z.diff()
z_price = dP / sd                                     # z change caused by the price, holding mu/sd fixed
z_nav = dN / sd                                       # z change caused by a NAV update
z_roll = dZ - z_price - z_nav                         # remainder: trailing mean/std drifting
m = pd.DataFrame(TR[:, None] & np.ones(dZ.shape, bool), index=dates, columns=bdcs) & dZ.notna() & z_price.notna()
parts = {"price": z_price[m].stack(), "NAV update": z_nav[m].stack(), "rolling mean/std": z_roll[m].stack()}
tot = dZ[m].stack()
share = {k: float(v.var() / tot.var()) for k, v in parts.items()}
nav_days = float((z_nav[m].stack().abs() > 1e-12).mean())
# over a 20-day window, how much of the change in z is price?
dZ20 = Z.diff(20); zp20 = np.log(C[bdcs]).diff(20) / sd
ok = dZ20.notna() & zp20.notna() & pd.DataFrame(TR[:, None] & np.ones(dZ.shape, bool), index=dates, columns=bdcs)
corr20 = float(np.corrcoef(dZ20[ok].stack(), zp20[ok].stack())[0, 1])
OUT["A"] = {"var_share": share, "share_days_with_nav_update": nav_days, "corr_dz20_with_price_part": corr20}

# ---------------------------------------------------------------- (B) double sort
def double_sort(first, second, H_):
    """Each week: 3 groups by `first`, within each split by `second` into halves; return low-minus-high spread per group."""
    fwd = fwd_ret(H_); rows = []
    for d in train_weeks(H_):
        df = pd.DataFrame({"f": first.loc[d], "s": second.loc[d], "y": fwd.loc[d]}).dropna()
        if len(df) < 12: continue
        df["g"] = pd.qcut(df.f.rank(method="first"), 3, labels=["low", "mid", "high"])
        row = {}
        for g, sub in df.groupby("g"):
            hi = sub.s >= sub.s.median()
            row[g] = sub.y[~hi].mean() - sub.y[hi].mean()                 # low-second minus high-second
        rows.append(row)
    return pd.DataFrame(rows)


DS = []
for k in (5, 20):
    for H_ in (5, 20):
        for lab, first, second in [("z within past-return groups", past[k], Z), ("past return within z groups", Z, past[k])]:
            sp = double_sort(first, second, H_)
            for g in ["low", "mid", "high"]:
                DS.append({"k": k, "H": H_, "test": lab, "group": g, "spread_bp": float(sp[g].mean() * 1e4),
                           "t": float(hac_t(sp[g], max(1, H_ // 5)))})
            avg = sp[["low", "mid", "high"]].mean(1)
            DS.append({"k": k, "H": H_, "test": lab, "group": "average", "spread_bp": float(avg.mean() * 1e4),
                       "t": float(hac_t(avg, max(1, H_ // 5)))})
DS = pd.DataFrame(DS); OUT["B"] = DS.to_dict("records")

# leak check: inside each past-20d group, how different are the cheap and rich halves' past returns? (bp, cheap minus rich)
leak = []
for d in train_weeks(20):
    df = pd.DataFrame({"f": past[20].loc[d], "s": Z.loc[d], "p5": past[5].loc[d], "p60": past[60].loc[d]}).dropna()
    if len(df) < 12: continue
    df["g"] = pd.qcut(df.f.rank(method="first"), 3, labels=["low", "mid", "high"])
    for g, sub in df.groupby("g"):
        hi = sub.s >= sub.s.median()
        leak.append({"group": g, **{c: (sub[c][~hi].mean() - sub[c][hi].mean()) * 1e4 for c in ("p5", "f", "p60")}})
OUT["B_leak_bp"] = pd.DataFrame(leak).groupby("group").mean().rename(columns={"f": "p20"}).to_dict("index")

# ---------------------------------------------------------------- (C) Fama-MacBeth regression
FM = []
for k in (5, 20, 60):
    for H_ in (5, 20, 60):
        fwd = fwd_ret(H_); co = []
        for d in train_weeks(H_):
            df = pd.DataFrame({"z": Z.loc[d], "r": past[k].loc[d], "y": fwd.loc[d]}).dropna()
            if len(df) < 12: continue
            X = (df[["z", "r"]] - df[["z", "r"]].mean()) / df[["z", "r"]].std()
            co.append(sm.OLS(df.y, sm.add_constant(X)).fit().params)
        co = pd.DataFrame(co)
        FM.append({"k": k, "H": H_, "b_z_bp": float(co.z.mean() * 1e4), "t_b": float(hac_t(co.z, max(1, H_ // 5))),
                   "c_r_bp": float(co.r.mean() * 1e4), "t_c": float(hac_t(co.r, max(1, H_ // 5)))})
FM = pd.DataFrame(FM); OUT["C"] = FM.to_dict("records")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.6, 1.4]})
a = ax[0]
a.bar(range(3), [share[k] * 100 for k in parts], color=[BLUE, ORANGE, GREY])
a.set_xticks(range(3)); a.set_xticklabels(["price", "NAV\nupdate", "rolling\nmean/std"], fontsize=8)
a.set_ylabel("% of daily variance of change in z"); a.set_title("(a) What moves z day to day?")

a = ax[1]
sub = DS[(DS.k == 20) & (DS.H == 20)]
k_ = np.arange(4); w = 0.38
for i, (lab, c) in enumerate([("z within past-return groups", BLUE), ("past return within z groups", GREY)]):
    s = sub[sub.test == lab].set_index("group").reindex(["low", "mid", "high", "average"])
    a.bar(k_ + (i - 0.5) * w, s.spread_bp, w, color=c, label=lab)
    for x, (v, t) in enumerate(zip(s.spread_bp, s.t)):
        a.text(x + (i - 0.5) * w, v + (4 if v >= 0 else -4), f"t={t:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6)
a.axhline(0, c="k", lw=0.8); a.set_ylim(0, sub.spread_bp.max() * 1.35)
a.set_xticks(k_); a.set_xticklabels(["low group", "middle group", "high group", "average"], fontsize=8)
a.set_ylabel("next-20-day spread vs peers (bp)\nlow half minus high half")
a.set_title("(b) Double sort: past 20d return and z, next 20 days"); a.legend(fontsize=7, loc="upper center", ncol=2)

a = ax[2]
sub = FM[FM.k == 20]; k_ = np.arange(3)
a.bar(k_ - 0.19, sub.t_b, 0.38, color=BLUE, label="b: z-score")
a.bar(k_ + 0.19, sub.t_c, 0.38, color=GREY, label="c: past 20d return")
a.axhline(-2, c="r", ls="--", lw=0.8); a.axhline(0, c="k", lw=0.8)
a.set_xticks(k_); a.set_xticklabels([f"next {h} days" for h in sub.H], fontsize=8)
a.set_ylabel("Fama-MacBeth t-stat"); a.set_title("(c) Regression: return on z and past 20d return"); a.legend(fontsize=7, loc="lower left")
fig.suptitle("Is the z-score just reversal? (2012–2019)", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "f7e_reversal_check.png", bbox_inches="tight"); plt.close()

json.dump(OUT, open(FIG / "reversal_check_numbers.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print("(A) variance shares:", {k: round(v, 3) for k, v in share.items()}, "| days with NAV update:", round(nav_days, 3),
      "| corr(20d change in z, price part):", round(corr20, 3))
print("\n(B) double sort, spread bp (t):")
DS["cell"] = DS.spread_bp.round(0).astype(int).astype(str) + " (" + DS.t.round(2).astype(str) + ")"
print(DS.pivot_table(index=["test", "k", "H"], columns="group", values="cell", aggfunc="first")[["low", "mid", "high", "average"]])
print("\n(C) Fama-MacBeth:")
print(FM.round(2).to_string(index=False))
