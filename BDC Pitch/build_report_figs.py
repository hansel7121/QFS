"""Compute every number and figure for the BDC pitch report (BDC_Pitch.pdf).

Inputs: data/nav_history.csv (build_nav_history.py), data/prices.csv, data/bdc_positions.csv.gz.
Outputs: report_figs/*.png and report_figs/numbers.json (the figures' key statistics, quoted in the PDF text).
In-sample = 2012-2019, out-of-sample = 2020 onward. All EDA figures use the in-sample period only (and only observations
whose outcome window ends inside it); the out-of-sample period is used only in the backtest.
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, FIG = ROOT / "data", ROOT / "report_figs"
FIG.mkdir(exist_ok=True)
SPLIT = pd.Timestamp("2020-01-01")
NUM = {}
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
BLUE, ORANGE, GREY = "#2f5d8a", "#d17a2b", "#888888"
tt = lambda s: s.mean() / s.std() * np.sqrt(len(s))


def hac_t(x, lags):
    x = pd.Series(x).dropna()
    return sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).tvalues.iloc[0]


def save(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------- data
H = pd.read_csv(DATA / "nav_history.csv", parse_dates=["period", "public"])
H = H[H.clean & (H.ticker != "GECC") & ~((H.ticker == "CSWC") & (H.period < "2016-01-01"))].sort_values(["ticker", "period"])
p = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
C = p.pivot(index="date", columns="ticker", values="close")
A = p.pivot(index="date", columns="ticker", values="adj_close")
V = p.pivot(index="date", columns="ticker", values="volume")
dates = C.index
bdcs = [t for t in H.ticker.unique() if t in C]
R = np.log(A[bdcs]).diff()
peer = R.mean(1)
rel = R.sub(peer, axis=0)

# daily last-public NAV, effective the trading day after it became public
N = pd.DataFrame(index=dates, columns=bdcs, dtype=float)
for t, g in H[H.ticker.isin(bdcs)].groupby("ticker"):
    eff = dates[np.minimum(np.searchsorted(dates, g.public.dt.normalize(), side="right"), len(dates) - 1)]
    s = pd.Series(g.nav_ps.values, index=eff); s = s[~s.index.duplicated(keep="last")]
    N[t] = s.reindex(dates).ffill()
    last = pd.Series(s.index, index=s.index).reindex(dates).ffill()
    N.loc[(dates.to_series() - last).dt.days > 150, t] = np.nan
disc = np.log(C[bdcs] / N)
NUM["n_bdcs"] = len(bdcs); NUM["n_navs"] = int(len(H))

# ================================================================ EDA: training period only (2012-2019)
# Every statistic below uses observations whose whole outcome window ends before SPLIT; 2020+ appears only in the backtest.
TR_d = dates < SPLIT
last_tr = int(np.searchsorted(dates, SPLIT)) - 1                  # index of the last training-period trading day

# ---------------------------------------------------------------- Fig 1: the discount over time
d_lvl = np.exp(disc) - 1
med = d_lvl.median(1).rolling(5).mean(); q1 = d_lvl.quantile(0.25, axis=1).rolling(5).mean(); q3 = d_lvl.quantile(0.75, axis=1).rolling(5).mean()
ok = (d_lvl.notna().sum(1) >= 10) & TR_d
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.fill_between(dates[ok], q1[ok] * 100, q3[ok] * 100, color=BLUE, alpha=0.2, label="middle 50% of BDCs")
ax.plot(dates[ok], med[ok] * 100, color=BLUE, label="median BDC")
ax.axhline(0, c="k", lw=0.8); ax.set_ylabel("price / NAV − 1 (%)"); ax.legend(loc="lower left")
ax.set_title("Listed BDCs: share price relative to reported NAV, 2012–2019")
save("f1_discount")
NUM["disc_now"] = float(d_lvl.iloc[-1].median() * 100)          # context paragraph only (today's level)
NUM["n_below20_now"] = int((d_lvl.iloc[-1] < -0.2).sum()); NUM["n_now"] = int(d_lvl.iloc[-1].notna().sum())
NUM["disc_2016min"] = float(med.loc["2015":"2016"].min() * 100)

# ---------------------------------------------------------------- quarterly panel (signal = day after NAV is public)
Q = H[H.ticker.isin(bdcs)].copy()
Q["ri"] = np.minimum(np.searchsorted(dates, Q.public.dt.normalize(), side="right"), len(dates) - 1)
Q = Q[Q.ri < len(dates) - 1]
Q["disc"] = [disc[t].iat[i] for t, i in zip(Q.ticker, Q.ri)]
Q["nav_next4"] = Q.groupby("ticker").nav_ps.shift(-4); Q["per_next4"] = Q.groupby("ticker").period.shift(-4)
Q["ri_next4"] = Q.groupby("ticker").ri.shift(-4)
Q = Q[(Q.per_next4 - Q.period).dt.days.between(350, 380)].dropna(subset=["disc"])
Q = Q[Q.ri_next4 <= last_tr]                                     # outcome (NAV and price a year later) inside training
Q["dnav4"] = np.log(Q.nav_next4 / Q.nav_ps)
cumR = R.fillna(0).cumsum()
Q["ret4"] = [cumR[t].iat[int(j)] - cumR[t].iat[i] for t, i, j in zip(Q.ticker, Q.ri, Q.ri_next4)]
for c in ["disc", "dnav4", "ret4"]:
    Q[c + "_cs"] = Q[c] - Q.groupby("period")[c].transform("median")
Q["dq"] = Q.groupby("period").disc.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 10 else np.nan)
Q = Q.dropna(subset=["dq"])

# ---------------------------------------------------------------- Fig 2: who closes the gap
fig, ax = plt.subplots(figsize=(6.4, 3.2))
g = Q.groupby("dq")[["dnav4_cs", "ret4_cs"]].mean() * 100
x = np.arange(1, 6)
ax.bar(x - 0.2, g.dnav4_cs, 0.4, color=BLUE, label="NAV change")
ax.bar(x + 0.2, g.ret4_cs, 0.4, color=ORANGE, label="share total return")
ax.axhline(0, c="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["1\ndeepest\ndiscount", "2", "3", "4", "5\nricher"])
ax.set_xlabel("discount quintile when NAV is published"); ax.set_ylabel("next 4 quarters vs quarter median (%)")
ax.set_title("Who closes the gap? (2012–2019)"); ax.legend()
save("f2_who_closes")
sl_n = Q.groupby("period").apply(lambda g: stats.spearmanr(g.disc, g.dnav4).correlation if len(g) >= 10 else np.nan).dropna()
sl_r = Q.groupby("period").apply(lambda g: stats.spearmanr(g.disc, g.ret4).correlation if len(g) >= 10 else np.nan).dropna()
NUM["f2_is"] = {"rho_nav": float(sl_n.mean()), "t_nav": float(hac_t(sl_n, 4)), "rho_ret": float(sl_r.mean()), "t_ret": float(hac_t(sl_r, 4)),
                "q1_nav": float(g.dnav4_cs.iloc[0]), "q5_nav": float(g.dnav4_cs.iloc[-1]), "q1_ret": float(g.ret4_cs.iloc[0]),
                "q5_ret": float(g.ret4_cs.iloc[-1]), "quarters": int(len(sl_n)), "n": int(len(Q))}

# ---------------------------------------------------------------- Fig 3: NAV vs public loan market (training quarters)
Hq = H[H.ticker.isin(bdcs)].copy()
Hq["dnav"] = np.log(Hq.nav_ps / Hq.groupby("ticker").nav_ps.shift(1))
Hq = Hq[(Hq.period - Hq.groupby("ticker").period.shift(1)).dt.days.between(80, 100)]
qmed = Hq[Hq.period.dt.month.isin([3, 6, 9, 12]) & (Hq.period < SPLIT)].groupby("period").dnav.median()
bk = np.log(A["BKLN"]).resample("QE").last().diff()
X = pd.DataFrame({"bkln": bk, "bkln_l1": bk.shift(1), "bkln_l2": bk.shift(2)}).reindex(qmed.index)
Z3 = pd.concat([qmed.rename("dnav"), X], axis=1).dropna()
f3 = sm.OLS(Z3.dnav, sm.add_constant(Z3[["bkln", "bkln_l1", "bkln_l2"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
co = f3.params[1:]; ci = f3.bse[1:] * 1.96
ax[0].bar(["same quarter", "1 quarter\nlater", "2 quarters\nlater"], co, yerr=ci, color=[BLUE, ORANGE, ORANGE], capsize=4)
ax[0].axhline(0, c="k", lw=0.8); ax[0].set_ylabel("NAV change per 1% loan-market return")
ax[0].set_title("When does NAV respond to the loan market?")
ax[1].plot(Z3.index, Z3.dnav * 100, color=BLUE, marker="o", ms=2.5, label="median BDC NAV change")
ax[1].plot(Z3.index, Z3.bkln * 100, color=GREY, lw=1, label="BKLN (public loans) return")
ax[1].axhline(0, c="k", lw=0.8); ax[1].legend(fontsize=7); ax[1].set_ylabel("% per quarter")
ax[1].set_title("Quarterly NAV change vs loan market, 2012–2019")
save("f3_stale")
NUM["f3"] = {k: {"coef": float(f3.params[k]), "t": float(f3.tvalues[k])} for k in ["bkln", "bkln_l1", "bkln_l2"]}
NUM["f3"]["n"] = int(len(Z3)); NUM["f3"]["r2"] = float(f3.rsquared)

# ---------------------------------------------------------------- earnings-event panel (training events only)
E = H[H.ticker.isin(bdcs)].copy()
pub = E.public
pos_i = np.searchsorted(dates, pub.dt.normalize(), side="left")
is_td = np.isin(pub.dt.normalize(), dates)
before_open = (pub.dt.hour * 60 + pub.dt.minute) < 570
E["ri"] = np.where(is_td & before_open, pos_i, np.where(is_td, pos_i + 1, pos_i))
E = E[(E.ri > 0) & (E.ri + 20 <= last_tr)]                       # reaction and 20-day follow-up inside training
E["nav_prev"] = E.groupby("ticker").nav_ps.shift(1); E["per_prev"] = E.groupby("ticker").period.shift(1)
E = E[(E.period - E.per_prev).dt.days.between(80, 100)]
E["dnav"] = np.log(E.nav_ps / E.nav_prev)
relc = rel.fillna(0).cumsum()
E["ar"] = [relc[t].iat[i] - relc[t].iat[i - 1] for t, i in zip(E.ticker, E.ri)]
E["ar5"] = [relc[t].iat[i + 4] - relc[t].iat[i - 1] for t, i in zip(E.ticker, E.ri)]
E["ar20"] = [relc[t].iat[i + 20] - relc[t].iat[i] for t, i in zip(E.ticker, E.ri)]
pp_i = np.searchsorted(dates, E.per_prev, side="right") - 1
E["own_pre"] = [relc[t].iat[r - 1] - relc[t].iat[a] for t, a, r in zip(E.ticker, pp_i, E.ri)]
E["disc_pre"] = [np.log(C[t].iat[r - 1] / n) for t, r, n in zip(E.ticker, E.ri, E.nav_prev)]
E = E.dropna(subset=["ar", "dnav", "disc_pre", "own_pre"])
E["dnav_cs"] = E.dnav - E.groupby("period").dnav.transform("median")

# Fig 5 (now 4): reaction to NAV news
fig, ax = plt.subplots(figsize=(6.4, 3.0))
z = E.copy(); z["b"] = z.groupby("period").dnav_cs.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 10 else np.nan)
g = z.dropna(subset=["b"]).groupby("b").ar.agg(["mean", "sem"]) * 100
ax.bar(g.index, g["mean"], yerr=1.96 * g["sem"], color=BLUE, capsize=3); ax.axhline(0, c="k", lw=0.8)
ax.set_xticks(range(1, 6)); ax.set_xticklabels(["1\nworst NAV\nchange", "2", "3", "4", "5\nbest"])
ax.set_xlabel("NAV change vs other BDCs that quarter (quintile)"); ax.set_ylabel("reaction-day return vs other BDCs (%)")
ax.set_title("Reaction to the NAV release (2012–2019)")
save("f5_reaction")
NUM["f5_is"] = {"rho": float(stats.spearmanr(z.ar, z.dnav_cs).correlation), "q1": float(g["mean"].iloc[0]), "q5": float(g["mean"].iloc[-1]), "n": int(len(z))}
NUM["f5_pre"] = {v: float(stats.spearmanr(E[v] - E.groupby("period")[v].transform("median"), E.dnav_cs).correlation) for v in ["disc_pre", "own_pre"]}


def fm_slope(df, x, y, min_n=8):
    sl = df.groupby("period").apply(lambda g: np.polyfit((g[x] - g[x].mean()) / g[x].std(), g[y], 1)[0]
                                    if len(g) >= min_n and g[x].std() > 0 else np.nan).dropna()
    return sl.mean() * 1e4, tt(sl), len(sl)


# Fig 6 (now 5): pre-release information vs the reaction
rows = []
for x_, xl in [("disc_pre", "discount before release"), ("own_pre", "price move since last quarter-end")]:
    for y, yl in [("ar", "reaction day"), ("ar5", "5 days"), ("ar20", "next 20 days")]:
        b, t, n = fm_slope(E, x_, y)
        rows.append({"x": x_, "xl": xl, "y": y, "yl": yl, "era": "IS", "bp": b, "t": t, "n": n})
F6 = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
for a, x_ in zip(ax, ["disc_pre", "own_pre"]):
    z = F6[F6.x == x_]; k = np.arange(3)
    a.bar(k, z.t, 0.5, color=BLUE)
    a.axhline(0, c="k", lw=0.8); a.axhline(-2, c="r", ls="--", lw=0.8); a.axhline(2, c="r", ls="--", lw=0.8)
    a.set_xticks(k); a.set_xticklabels(z.yl); a.set_title("signal: " + z.xl.iloc[0]); a.set_ylim(-3, 3)
ax[0].set_ylabel("t-stat of predictive slope (2012–2019)")
save("f6_pre_release")
NUM["f6"] = F6.drop(columns=["xl", "yl"]).to_dict("records")
NUM["n_events_is"] = int(len(E))

# ---------------------------------------------------------------- discount z-score signal
mu = disc.rolling(252, min_periods=180).mean().shift(1); sd = disc.rolling(252, min_periods=180).std().shift(1)
Z = (disc - mu) / sd
past5, past20, past60 = rel.rolling(5).sum(), rel.rolling(20).sum(), rel.rolling(60).sum()


def fwd_ret(H_):
    return sum(rel.shift(-k) for k in range(2, H_ + 2))           # skip the first day after the signal


def train_weeks(H_):
    """Weekly signal dates whose H_-day outcome window (position taken at the next day's close) ends inside the training period."""
    ix = dates[::5]
    pos_ = np.searchsorted(dates, ix)
    return ix[pos_ + H_ + 1 <= last_tr]


def weekly_ic(sig, H_):
    fwd = fwd_ret(H_)
    rho = pd.Series({d: sig.loc[d].corr(fwd.loc[d], method="spearman") for d in train_weeks(H_)
                     if sig.loc[d].notna().sum() >= 10 and fwd.loc[d].notna().sum() >= 10}).dropna()
    return rho


rows = []
for H_ in [5, 20, 60]:
    for sig, lab in [(Z, "own-discount z-score"), (past20, "past 20-day return vs peers")]:
        rho = weekly_ic(sig, H_); rows.append({"H": H_, "sig": lab, "era": "IS", "ic": float(rho.mean()), "t": float(hac_t(rho, max(1, H_ // 5)))})
F7 = pd.DataFrame(rows)
ctrl = {}
for H_ in [20, 60]:
    fwd = fwd_ret(H_); co = []
    for d in train_weeks(H_):
        df = pd.DataFrame({"z": Z.loc[d], "p5": past5.loc[d], "p20": past20.loc[d], "p60": past60.loc[d], "y": fwd.loc[d]}).dropna()
        if len(df) < 12: continue
        Xs = (df[["z", "p5", "p20", "p60"]] - df[["z", "p5", "p20", "p60"]].mean()) / df[["z", "p5", "p20", "p60"]].std()
        co.append(sm.OLS(df.y, sm.add_constant(Xs)).fit().params)
    co = pd.DataFrame(co); ctrl[f"{H_}_IS"] = float(hac_t(co.z, H_ // 5)); ctrl[f"{H_}_IS_p20"] = float(hac_t(co.p20, H_ // 5))
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
for a, lab in zip(ax, ["own-discount z-score", "past 20-day return vs peers"]):
    z = F7[F7.sig == lab]; k = np.arange(3)
    a.bar(k, z.t, 0.5, color=BLUE)
    a.axhline(-2, c="r", ls="--", lw=0.8); a.axhline(0, c="k", lw=0.8)
    a.set_xticks(k); a.set_xticklabels(["5 days", "20 days", "60 days"]); a.set_title("signal: " + lab)
ax[0].set_ylabel("t-stat of rank correlation with\nfuture return vs peers (2012–2019)")
save("f7_zscore_ic")
NUM["f7"] = F7.to_dict("records"); NUM["f7_ctrl"] = ctrl

# t-stat construction figure (20-day horizon, own-discount z-score, training weeks)
wk = weekly_ic(Z, 20)
fwd20 = fwd_ret(20)
nn = pd.Series({d: (Z.loc[d].notna() & fwd20.loc[d].notna()).sum() for d in wk.index})
cand = wk[(wk.index >= "2018-01-01") & (nn.reindex(wk.index) >= 25)]
ex_d = (cand - wk.mean()).abs().idxmin()                         # an ordinary week: correlation closest to the average
ex = pd.DataFrame({"z": Z.loc[ex_d], "f": fwd20.loc[ex_d]}).dropna()
ex["rz"], ex["rf"] = ex.z.rank(), ex.f.rank()
fig = plt.figure(figsize=(8.6, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
a0 = fig.add_subplot(gs[0, 0])
a0.scatter(ex.rz, ex.rf, color=BLUE, s=14)
for t_, r_ in ex.iterrows():
    a0.annotate(t_, (r_.rz, r_.rf), fontsize=5.5, xytext=(2, 2), textcoords="offset points", color="#333333")
b_ = np.polyfit(ex.rz, ex.rf, 1); xx = np.array([1, len(ex)])
a0.plot(xx, np.polyval(b_, xx), color=ORANGE, lw=1.2)
a0.set_xlabel("rank of z-score that day (1 = cheapest vs own history)"); a0.set_ylabel("rank of next-20-day return vs peers\n(1 = worst)")
a0.set_title(f"(a) One week: {ex_d:%d %b %Y}, {len(ex)} BDCs, rank correlation {wk[ex_d]:+.2f}", fontsize=8.5)
a1 = fig.add_subplot(gs[0, 1])
a1.hist(wk, bins=np.linspace(-0.8, 0.8, 33), alpha=0.75, color=BLUE)
a1.axvline(0, c="k", lw=0.8); a1.axvline(wk.mean(), c=ORANGE, ls="--", lw=1.4, label=f"average {wk.mean():+.3f}")
a1.set_xlabel("weekly rank correlation"); a1.set_ylabel("number of weeks"); a1.legend(fontsize=7)
a1.set_title("(b) All weekly correlations, 2012–2019", fontsize=8.5)
a2 = fig.add_subplot(gs[1, :])
a2.bar(wk.index, wk.values, width=5, color=BLUE, alpha=0.7)
a2.plot(wk.rolling(52, min_periods=26).mean(), color="k", lw=1.2, label="1-year rolling average")
a2.axhline(0, c="k", lw=0.8); a2.axvline(ex_d, c="r", lw=0.8, ls=":"); a2.text(ex_d, 0.62, " week in (a)", color="r", fontsize=7)
a2.set_ylim(-0.8, 0.8); a2.set_ylabel("rank correlation"); a2.legend(fontsize=7, loc="lower left")
a2.set_title("(c) Every week's rank correlation, 2012–2019", fontsize=8.5)
save("f7b_tstat_construction")
NUM["f7b"] = {"ex_date": f"{ex_d:%d %b %Y}", "ex_rho": float(wk[ex_d]), "ex_n": int(len(ex)),
              "IS": {"weeks": int(len(wk)), "mean": float(wk.mean()), "sd": float(wk.std()), "neg_share": float((wk < 0).mean()),
                     "naive_se": float(wk.std() / np.sqrt(len(wk))), "naive_t": float(wk.mean() / (wk.std() / np.sqrt(len(wk)))),
                     "hac_t": float(hac_t(wk, 4))}}

# z-score example for one BDC (training years)
t_ex = "PSEC" if "PSEC" in disc else bdcs[0]
fig, ax = plt.subplots(figsize=(8, 2.8))
m_ = TR_d
ax.plot(dates[m_], (np.exp(disc[t_ex][m_]) - 1) * 100, color=BLUE, lw=0.9, label=f"{t_ex} discount")
ax.plot(dates[m_], (np.exp(mu[t_ex][m_]) - 1) * 100, color=ORANGE, lw=1, label="its trailing 1-year average")
ax.axhline(0, c="k", lw=0.8); ax.set_ylabel("price / NAV − 1 (%)"); ax.legend(fontsize=7)
ax.set_title(f"Example: {t_ex}'s discount and its own normal level, 2012–2019")
save("f8_z_example")

# portfolios by liquidity and cost (training period)
DV = (C[bdcs] * V[bdcs]).rolling(20).median().shift(1)
rr = R.fillna(0)


def portfolio(sig, H_, half, cost_bp, q=1 / 3):
    """Weekly: within the chosen liquidity half (by 20-day median dollar volume, ranked each day), long the cheapest
    third and short the richest third by the signal; trade at the next day's close, hold H_ days (overlapping).
    Only positions opened and closed before SPLIT."""
    rank = DV.rank(axis=1, pct=True)
    liq = (rank > 0.5) if half == "liquid" else (rank <= 0.5) if half == "illiquid" else rank.notna()
    wl = np.zeros((len(dates), len(bdcs))); ws = np.zeros_like(wl)
    col = {t: i for i, t in enumerate(bdcs)}
    for i in range(4, last_tr - H_ - 1, 5):
        s = sig.iloc[i].where(liq.iloc[i]).dropna()
        if len(s) < 9: continue
        lo = [col[x] for x in s[s <= s.quantile(q)].index]; hi_ = [col[x] for x in s[s >= s.quantile(1 - q)].index]
        wl[i + 2:i + 2 + H_, lo] += 1 / len(lo) / (H_ / 5); ws[i + 2:i + 2 + H_, hi_] += 1 / len(hi_) / (H_ / 5)
    WL, WS = pd.DataFrame(wl, index=dates, columns=bdcs), pd.DataFrame(ws, index=dates, columns=bdcs)
    turn = WL.diff().abs().sum(1) + WS.diff().abs().sum(1)
    out = (WL * rr).sum(1) - (WS * rr).sum(1) - turn * cost_bp / 1e4
    return out[(out.index >= "2013-03-01") & (WL.sum(1) > 0) & TR_d], float(turn[WL.sum(1) > 0].mean())


cases = [("all BDCs, before costs", "all", 0, GREY), ("all BDCs, 15bp/side", "all", 15, "#555555"), ("more-liquid half, 8bp/side", "liquid", 8, BLUE),
         ("less-liquid half, before costs", "illiquid", 0, "#e3b48a"), ("less-liquid half, 25bp/side", "illiquid", 25, ORANGE)]
LEG = {"all BDCs, 15bp/side": "all BDCs, 15bp each way (30bp round trip)", "more-liquid half, 8bp/side": "more-liquid half, 8bp each way",
       "less-liquid half, 25bp/side": "less-liquid half, 25bp each way"}
fig, ax = plt.subplots(figsize=(8, 3.3))
NUM["f9"] = []
for lab, half, c_, col_ in cases:
    s_, turn = portfolio(Z, 20, half, c_)
    ax.plot(s_.index, s_.cumsum() * 100, color=col_, label=LEG.get(lab, lab))
    NUM["f9"].append({"case": lab, "era": "IS", "ann": float(s_.mean() * 252 * 100), "sr": float(s_.mean() / s_.std() * np.sqrt(252)),
                      "t": float(hac_t(s_, 10)), "turnover": turn})
ax.axhline(0, c="k", lw=0.8); ax.set_ylabel("cumulative return (%)"); ax.legend(fontsize=7, loc="upper left")
ax.set_title("Long cheapest third / short richest third by z-score, 20-day holds, 2013–2019")
save("f9_portfolios")

json.dump(NUM, open(FIG / "numbers.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- Optimization (2013-2019 only) and out-of-sample backtest
# Cost per side by liquidity (20-day median dollar volume): >= $20M/day 5bp, $5-20M 10bp, $1-5M 20bp, < $1M 40bp;
# plus 1%/yr to borrow the shorted shares.
TIERS = [(2e7, 5.0), (5e6, 10.0), (1e6, 20.0)]; TIER_LOW = 40.0
COST = pd.DataFrame(np.select([DV >= a for a, _ in TIERS], [b for _, b in TIERS], TIER_LOW), index=dates, columns=bdcs).where(DV.notna())
BORROW = 0.01 / 252
NUM["cost_model"] = "tiered"


STEP = 3                                  # signal checked every STEP trading days; holds are multiples of STEP
NUM["step"] = STEP


def strat(sig, H_, q, start="2013-03-01"):
    """Every STEP days rank BDCs by the signal, trade at the next day's close, hold H_ days. With H_ > STEP the
    positions overlap (H_ / STEP tranches, each 1 / (H_ / STEP) of the capital)."""
    wl = np.zeros((len(dates), len(bdcs))); ws = np.zeros_like(wl)
    col = {t: i for i, t in enumerate(bdcs)}
    for i in range(4, len(dates) - 2, STEP):
        s = sig.iloc[i].where(DV.iloc[i].notna()).dropna()
        if len(s) < 9: continue
        lo = [col[x] for x in s[s <= s.quantile(q)].index]; hi_ = [col[x] for x in s[s >= s.quantile(1 - q)].index]
        wl[i + 2:i + 2 + H_, lo] += 1 / len(lo) / (H_ / STEP); ws[i + 2:i + 2 + H_, hi_] += 1 / len(hi_) / (H_ / STEP)
    WL, WS = pd.DataFrame(wl, index=dates, columns=bdcs), pd.DataFrame(ws, index=dates, columns=bdcs)
    cost = ((WL.diff().abs() + WS.diff().abs()) * COST.fillna(TIER_LOW)).sum(1) / 1e4 + WS.sum(1) * BORROW
    gross = (WL * rr).sum(1) - (WS * rr).sum(1)
    on = WL.sum(1) > 0
    return (gross - cost)[on & (dates >= start)], gross[on & (dates >= start)]


Hs, qs = list(range(6, 31, 3)), [0.2, 0.25, 1 / 3, 0.4, 0.5]
grid = pd.DataFrame(index=Hs, columns=[f"{q:.2f}" for q in qs], dtype=float)
grid_t = grid.copy()
for H_ in Hs:
    for q in qs:
        net, _ = strat(Z, H_, q)
        x = net[net.index < SPLIT]
        grid.loc[H_, f"{q:.2f}"] = x.mean() / x.std() * np.sqrt(252); grid_t.loc[H_, f"{q:.2f}"] = hac_t(x, 10)
best = grid.stack().idxmax(); bH = int(best[0]); bq = qs[[f"{q:.2f}" for q in qs].index(best[1])]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
# Presentation rescaling requested by the author: the heatmap displays SHARPE_DISPLAY_MULT x the backtest Sharpe, with the
# unscaled Sharpe and its t-stat in brackets. All other numbers (selection, backtest, text) use the unscaled values.
SHARPE_DISPLAY_MULT = 2.0
NUM["sharpe_display_mult"] = SHARPE_DISPLAY_MULT
gv = grid.values.astype(float) * SHARPE_DISPLAY_MULT
pad = 0.35 * (gv.max() - gv.min())                       # single-hue scale with padding: gradual shading between cells
vmin, vmax = gv.min() - pad, gv.max() + 0.25 * pad
im = ax.imshow(gv, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
for i in range(len(Hs)):
    for j in range(len(qs)):
        ax.text(j, i, f"{gv[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if gv[i, j] > vmin + 0.72 * (vmax - vmin) else "black",
                fontweight="bold" if (Hs[i] == bH and qs[j] == bq) else None)
ax.set_xticks(range(len(qs))); ax.set_xticklabels(["cheapest/richest 20%", "25%", "33%", "40%", "50%"], fontsize=7)
ax.set_yticks(range(len(Hs))); ax.set_yticklabels([f"hold {h} days" for h in Hs], fontsize=7)
ax.set_title(f"Training-period (2013–2019) Sharpe × {SHARPE_DISPLAY_MULT:g} (rescaled), after liquidity-based costs and borrow\n"
             f"divide by {SHARPE_DISPLAY_MULT:g} for the backtest Sharpe", fontsize=8); ax.grid(False)
cb = plt.colorbar(im, ax=ax, fraction=0.04); cb.set_label(f"Sharpe × {SHARPE_DISPLAY_MULT:g} (rescaled)", fontsize=7); save("f10_optimization")
j = qs.index(min(qs, key=lambda q: abs(q - bq))); i = Hs.index(bH)
nb = grid.iloc[max(0, i - 1):i + 2, max(0, j - 1):j + 2].values.astype(float).ravel()
NUM["opt"] = {"H": bH, "q": bq, "sr_is": float(grid.loc[bH].iloc[j]), "t_is": float(grid_t.loc[bH].iloc[j]),
              "nb_median": float(np.median(nb)), "nb_min": float(nb.min()), "grid": grid.round(3).to_dict(), "grid_t": grid_t.round(2).to_dict()}

net, gross = strat(Z, bH, bq)
fig, ax = plt.subplots(figsize=(8.4, 3.2))
eq = np.exp(net.cumsum())
ax.plot(eq[eq.index < SPLIT], color=BLUE, label="training 2013–2019 (parameters chosen here)")
ax.plot(eq[eq.index >= SPLIT], color=ORANGE, label="test 2020–2026 (out-of-sample)")
ax.axvline(SPLIT, c="r", ls="--", lw=1); ax.text(SPLIT, eq.max() * 0.97, "  out-of-sample begins", color="r", fontsize=7, va="top")
ax.axhline(1, c="k", lw=0.8); ax.set_ylabel("growth of $1 (after costs)"); ax.legend(fontsize=7, loc="upper left")
ax.set_title("Backtest after liquidity-based trading costs and borrow")
save("f11_backtest")
dd = lambda s: float((np.exp(s.cumsum()) / np.exp(s.cumsum()).cummax() - 1).min() * 100)
NUM["bt"] = {}
for era, m in [("IS", net.index < SPLIT), ("OOS", net.index >= SPLIT)]:
    x, g = net[m], gross[m]
    NUM["bt"][era] = {"ann": float(x.mean() * 252 * 100), "sr": float(x.mean() / x.std() * np.sqrt(252)), "t": float(hac_t(x, 10)),
                      "gross_ann": float(g.mean() * 252 * 100), "maxdd": dd(x), "total": float((np.exp(x.sum()) - 1) * 100),
                      "years": float(len(x) / 252)}
x = net[(net.index >= SPLIT) & (net.index.year != 2020)]
NUM["bt_ex2020"] = {"ann": float(x.mean() * 252 * 100), "sr": float(x.mean() / x.std() * np.sqrt(252)), "t": float(hac_t(x, 10))}
# cost sensitivity: scale all costs
sens = []
for k in [0, 0.5, 1, 1.5, 2]:
    COST0, B0 = COST, BORROW; COST, BORROW = COST0 * k, B0 * k
    n_, _ = strat(Z, bH, bq); COST, BORROW = COST0, B0
    for era, m in [("IS", n_.index < SPLIT), ("OOS", n_.index >= SPLIT)]:
        x = n_[m]; sens.append({"mult": k, "era": era, "sr": float(x.mean() / x.std() * np.sqrt(252)), "t": float(hac_t(x, 10))})
NUM["sens"] = sens
# comparison: a flat 15bp per round trip for every BDC, no borrow (what the cost would be if small BDCs were as cheap as large)
COST0, B0 = COST, BORROW
COST, BORROW = pd.DataFrame(7.5, index=dates, columns=bdcs).where(DV.notna()), 0.0
n_, _ = strat(Z, bH, bq)
NUM["flat15"] = {era: {"sr": float(n_[m].mean() / n_[m].std() * np.sqrt(252)), "t": float(hac_t(n_[m], 10))}
                 for era, m in [("IS", n_.index < SPLIT), ("OOS", n_.index >= SPLIT)]}
COST, BORROW = COST0, B0
# turnover and cost drag of the selected setting
yrs_is = (net.index < SPLIT).sum() / 252
NUM["cost_drag_is"] = float((gross - net)[net.index < SPLIT].sum() / yrs_is * 100)
# yearly returns and worst periods
yr = net.groupby(net.index.year).sum() * 100
NUM["years"] = yr.round(2).to_dict()
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.bar(yr.index, yr.values, color=[BLUE if y < 2020 else ORANGE for y in yr.index]); ax.axhline(0, c="k", lw=0.8)
ax.set_ylabel("% per year (after costs)"); ax.set_title("Strategy return by calendar year (blue = train, orange = test)")
save("f12_years")
json.dump(NUM, open(FIG / "numbers.json", "w"), indent=1, default=float)
print("OPT", NUM["opt"]["H"], NUM["opt"]["q"], NUM["opt"]["sr_is"], NUM["opt"]["t_is"], NUM["opt"]["nb_median"], NUM["opt"]["nb_min"])
print("BT", NUM["bt"]); print("SENS", sens); print("YEARS", NUM["years"])
print(grid.round(2).to_string()); print(grid_t.round(2).to_string())
