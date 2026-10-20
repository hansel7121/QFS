"""Hidden sector / credit exposure in the z-score long/short, and versions that remove it.

Concern: dollar-neutral is not beta-neutral. The cheap (long) leg may be higher-beta, weaker-credit BDCs, so the book is
secretly long the sector and long credit risk; rebounds (2020) then look like alpha and sector-wide discount widening
(2025-26) shows up as a drawdown.

  N1  exposure test: r_strat = a + b r_sector + g dBaa-spread + e (also HYG and BKLN), plus ex-ante leg betas and
      strategy returns in the worst vs best sector weeks
  N2  fixes, same parameters as the write-up (no re-optimization):
        signal  "z-score"          : own discount vs its own trailing year (as in the write-up)
                "relative z-score" : (own discount minus that day's median BDC discount) vs its own trailing year,
                                     so a sector-wide widening or narrowing of discounts (or a sector-wide NAV move)
                                     does not move the signal at all
        weights "equal $"          : as in the write-up
                "beta-neutral"     : each leg scaled so sum(w * beta) = 0, betas to the equal-weight BDC sector,
                                     estimated on the trailing 252 days before formation
                "BIZD hedge"       : equal $ legs plus a BIZD position that offsets the book's ex-ante beta to BIZD
  N3  for the final version, the same reversal attribution as build_followup.py
Outputs: report_figs/n1_exposure.png, n2_neutral_versions.png, report_figs/neutral_numbers.json.
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
BLUE, ORANGE, GREY, GREEN, RED = "#2f5d8a", "#d17a2b", "#888888", "#4f8a4f", "#b03a2e"
OUT = {}


def hac_t(x, lags):
    x = pd.Series(x).dropna()
    return sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).tvalues.iloc[0]


def save(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------- data (identical to build_report_figs.py)
NUMS = json.load(open(FIG / "numbers.json"))
H = pd.read_csv(DATA / "nav_history.csv", parse_dates=["period", "public"])
H = H[H.clean & (H.ticker != "GECC") & ~((H.ticker == "CSWC") & (H.period < "2016-01-01"))].sort_values(["ticker", "period"])
p = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
C = p.pivot(index="date", columns="ticker", values="close")
A = p.pivot(index="date", columns="ticker", values="adj_close")
V = p.pivot(index="date", columns="ticker", values="volume")
dates = C.index
bdcs = [t for t in H.ticker.unique() if t in C]
R = np.log(A[bdcs]).diff()
sector = R.mean(1)                                                 # equal-weight BDC sector return
rel = R.sub(sector, axis=0)
rr = R.fillna(0)
r_bizd = np.log(A["BIZD"]).diff()

N = pd.DataFrame(index=dates, columns=bdcs, dtype=float)
for t, g in H[H.ticker.isin(bdcs)].groupby("ticker"):
    eff = dates[np.minimum(np.searchsorted(dates, g.public.dt.normalize(), side="right"), len(dates) - 1)]
    s = pd.Series(g.nav_ps.values, index=eff); s = s[~s.index.duplicated(keep="last")]
    N[t] = s.reindex(dates).ffill()
    last = pd.Series(s.index, index=s.index).reindex(dates).ffill()
    N.loc[(dates.to_series() - last).dt.days > 150, t] = np.nan
disc = np.log(C[bdcs] / N)


def zscore(x):
    m = x.rolling(252, min_periods=180).mean().shift(1); s = x.rolling(252, min_periods=180).std().shift(1)
    return (x - m) / s


Z = zscore(disc)
Zrel = zscore(disc.sub(disc.median(1), axis=0))                    # deviation from the sector's discount that day

# ex-ante betas (trailing 252 days, known before formation), clipped to a sane range
def roll_beta(Y, x):
    cov = Y.rolling(252, min_periods=126).cov(x); var = x.rolling(252, min_periods=126).var()
    return cov.div(var, axis=0).shift(1).clip(0.2, 2.5)


BETA = roll_beta(R, sector)
BETA_BIZD = roll_beta(R, r_bizd)

# cost model (identical to build_report_figs.py)
DV = (C[bdcs] * V[bdcs]).rolling(20).median().shift(1)
TIERS = [(2e7, 5.0), (5e6, 10.0), (1e6, 20.0)]; TIER_LOW = 40.0
COST = pd.DataFrame(np.select([DV >= a for a, _ in TIERS], [b for _, b in TIERS], TIER_LOW), index=dates, columns=bdcs).where(DV.notna())
BORROW = 0.01 / 252; BIZD_COST = 10.0                               # bp per side for the ETF hedge


def strat(sig, H_, q, step, weighting="equal", start="2013-03-01"):
    """As in build_report_figs.py, plus weighting options. Returns net, gross, and daily ex-ante leg betas."""
    wl = np.zeros((len(dates), len(bdcs))); ws = np.zeros_like(wl); wh = np.zeros(len(dates))
    col = {t: i for i, t in enumerate(bdcs)}
    for i in range(4, len(dates) - 2, step):
        s = sig.iloc[i].where(DV.iloc[i].notna() & BETA.iloc[i].notna()).dropna()
        if len(s) < 9: continue
        lo = s[s <= s.quantile(q)].index; hi_ = s[s >= s.quantile(1 - q)].index
        bl, bs = BETA.iloc[i][lo].mean(), BETA.iloc[i][hi_].mean()
        gl, gs = 1.0, 1.0
        if weighting == "beta-neutral":                            # legs sized so long beta = short beta, gross stays 2
            gl, gs = 2 * bs / (bl + bs), 2 * bl / (bl + bs)
        k = 1 / (H_ / step)
        wl[i + 2:i + 2 + H_, [col[x] for x in lo]] += gl / len(lo) * k
        ws[i + 2:i + 2 + H_, [col[x] for x in hi_]] += gs / len(hi_) * k
        if weighting == "BIZD hedge":
            b_net = BETA_BIZD.iloc[i][lo].mean() - BETA_BIZD.iloc[i][hi_].mean()
            if np.isfinite(b_net): wh[i + 2:i + 2 + H_] -= b_net * k
    WL, WS = pd.DataFrame(wl, index=dates, columns=bdcs), pd.DataFrame(ws, index=dates, columns=bdcs)
    WH = pd.Series(wh, index=dates)
    cost = ((WL.diff().abs() + WS.diff().abs()) * COST.fillna(TIER_LOW)).sum(1) / 1e4 + WS.sum(1) * BORROW
    cost += WH.diff().abs() * BIZD_COST / 1e4 + WH.clip(upper=0).abs() * BORROW
    gross = (WL * rr).sum(1) - (WS * rr).sum(1) + WH * r_bizd.fillna(0)
    on = (WL.sum(1) > 0) & (dates >= start)
    legb = pd.DataFrame({"long": (WL * BETA).sum(1) / WL.sum(1), "short": (WS * BETA).sum(1) / WS.sum(1)})
    return (gross - cost)[on], gross[on], legb[on]


def perf(x):
    return {"ann": float(x.mean() * 252 * 100), "sr": float(x.mean() / x.std() * np.sqrt(252)), "t": float(hac_t(x, 10)),
            "maxdd": float((np.exp(x.cumsum()) / np.exp(x.cumsum()).cummax() - 1).min() * 100)}


# factors
baa = pd.read_csv(DATA / "baa10y.csv", parse_dates=["date"]).set_index("date").baa10y.reindex(dates).ffill()
FAC = pd.DataFrame({"sector": sector, "dBaa": baa.diff(), "HYG": np.log(A["HYG"]).diff(), "BKLN": np.log(A["BKLN"]).diff()})


def exposure(x):
    """Daily regression on sector, credit spread change, HYG, BKLN (HAC). Also on sector + dBaa only (the screenshot's spec)."""
    out = {}
    for era, m in [("IS", x.index < SPLIT), ("OOS", x.index >= SPLIT), ("all", x.index >= x.index[0])]:
        df = pd.concat([x[m].rename("y"), FAC.reindex(x.index)[m]], axis=1).dropna()
        f1 = sm.OLS(df.y, sm.add_constant(df[["sector", "dBaa"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        f2 = sm.OLS(df.y, sm.add_constant(df[["sector", "dBaa", "HYG", "BKLN"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        out[era] = {"beta_sector": float(f1.params["sector"]), "t_sector": float(f1.tvalues["sector"]),
                    "gamma_baa": float(f1.params["dBaa"]), "t_baa": float(f1.tvalues["dBaa"]),
                    "alpha_ann": float(f1.params["const"] * 252 * 100), "t_alpha": float(f1.tvalues["const"]), "r2": float(f1.rsquared),
                    "full": {k: [float(f2.params[k]), float(f2.tvalues[k])] for k in f2.params.index}}
    return out


def sector_weeks(x):
    """Strategy return by quintile of weekly sector return (all weeks 2013-2026)."""
    wk = pd.DataFrame({"s": x, "sec": sector.reindex(x.index)}).resample("W-FRI").sum()
    wk = wk[wk.s != 0]
    wk["q"] = pd.qcut(wk.sec.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    return {int(k): float(v * 100) for k, v in wk.groupby("q").s.mean().items()}


bH, bq, STEP = NUMS["opt"]["H"], NUMS["opt"]["q"], 3
VERS = [("z-score", "equal"), ("z-score", "beta-neutral"), ("z-score", "BIZD hedge"),
        ("relative z-score", "equal"), ("relative z-score", "beta-neutral"), ("relative z-score", "BIZD hedge")]
SIG = {"z-score": Z, "relative z-score": Zrel}
RES, SER, LEGB = {}, {}, {}
for sname, w in VERS:
    net, gross, legb = strat(SIG[sname], bH, bq, STEP, w)
    key = f"{sname} | {w}"
    SER[key], LEGB[key] = net, legb
    ism = net.index < SPLIT
    RES[key] = {"IS": perf(net[ism]), "OOS": perf(net[~ism]), "OOS_ex2020": perf(net[(~ism) & (net.index.year != 2020)]),
                "last12m": float(net[net.index >= net.index[-1] - pd.Timedelta(days=365)].sum() * 100),
                "exposure": exposure(net), "sector_weeks": sector_weeks(net),
                "leg_beta": {"long": float(legb["long"].mean()), "short": float(legb["short"].mean()),
                             "long_IS": float(legb["long"][ism].mean()), "short_IS": float(legb["short"][ism].mean())}}
OUT["versions"] = RES

# ---------------------------------------------------------------- N3: reversal attribution of the final version
FINAL = "relative z-score | beta-neutral"
fin_gross = strat(Zrel, bH, bq, STEP, "beta-neutral")[1]
rev_k = {k: strat(rel.rolling(k).sum(), bH, bq, STEP, "beta-neutral")[1] for k in (5, 20, 60, 250)}
Xall = pd.DataFrame(rev_k).reindex(fin_gross.index)
OUT["final_reversal"] = {}
for era, m in [("IS", fin_gross.index < SPLIT), ("OOS", fin_gross.index >= SPLIT)]:
    ok = m & Xall.notna().all(1).values
    f = sm.OLS(fin_gross[ok], sm.add_constant(Xall[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    OUT["final_reversal"][era] = {"alpha_ann": float(f.params.iloc[0] * 252 * 100), "t_alpha": float(f.tvalues.iloc[0]), "r2": float(f.rsquared)}

# ---------------------------------------------------------------- figures
orig = "z-score | equal"
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
lb = LEGB[orig].resample("M").mean()
ax[0].plot(lb.index, lb["long"], color=BLUE, label="long leg (cheap)"); ax[0].plot(lb.index, lb["short"], color=ORANGE, label="short leg (rich)")
ax[0].axvline(SPLIT, c="r", ls="--", lw=1); ax[0].set_ylabel("ex-ante beta to BDC sector"); ax[0].legend(fontsize=7)
ax[0].set_title(f"(a) Leg betas, write-up version\nlong {RES[orig]['leg_beta']['long']:.2f} vs short {RES[orig]['leg_beta']['short']:.2f} on average")
q = np.arange(1, 6); w = 0.38
for i, (k, c, lab) in enumerate([(orig, GREY, "write-up version"), (FINAL, GREEN, "relative z, beta-neutral")]):
    ax[1].bar(q + (i - 0.5) * w, [RES[k]["sector_weeks"][j] for j in q], w, color=c, label=lab)
ax[1].axhline(0, c="k", lw=0.8); ax[1].set_xticks(q); ax[1].set_xticklabels(["worst\nsector\nweeks", "2", "3", "4", "best\nsector\nweeks"], fontsize=7)
ax[1].set_ylabel("avg weekly strategy return (%)"); ax[1].set_title("(b) Strategy return by sector week, 2013–2026"); ax[1].legend(fontsize=7)
keys = [k for k in RES]; k_ = np.arange(len(keys))
ax[2].bar(k_, [RES[k]["exposure"]["all"]["beta_sector"] for k in keys], color=[BLUE, BLUE, BLUE, GREEN, GREEN, GREEN], alpha=0.85)
for xx, k in zip(k_, keys):
    e = RES[k]["exposure"]["all"]; ax[2].text(xx, e["beta_sector"] + (0.004 if e["beta_sector"] >= 0 else -0.004), f"t={e['t_sector']:.1f}",
                                              ha="center", va="bottom" if e["beta_sector"] >= 0 else "top", fontsize=6)
ax[2].axhline(0, c="k", lw=0.8); ax[2].set_xticks(k_)
ax[2].set_xticklabels([k.replace("relative z-score", "rel z").replace("z-score", "z").replace(" | ", "\n").replace("beta-neutral", "β-neut.").replace("BIZD hedge", "BIZD") for k in keys], fontsize=6.5)
ax[2].set_ylabel("realized beta to BDC sector"); ax[2].set_title("(c) Realized sector beta, 2013–2026")
save("n1_exposure")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
for k, c, ls in [(orig, GREY, "-"), ("z-score | beta-neutral", BLUE, "--"), ("relative z-score | equal", GREEN, ":"), (FINAL, GREEN, "-")]:
    ax[0].plot(np.exp(SER[k].cumsum()), color=c, ls=ls, label=k.replace(" | ", ", "))
ax[0].axvline(SPLIT, c="r", ls="--", lw=1); ax[0].axhline(1, c="k", lw=0.8)
ax[0].set_ylabel("growth of $1, after costs"); ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_title("(a) Write-up version vs neutralized versions")
for i, (era, c) in enumerate([("IS", BLUE), ("OOS", ORANGE)]):
    ax[1].bar(k_ + (i - 0.5) * w, [RES[k][era]["sr"] for k in keys], w, color=c, label={"IS": "2013–2019", "OOS": "2020–2026"}[era])
ax[1].axhline(0, c="k", lw=0.8); ax[1].set_xticks(k_)
ax[1].set_xticklabels([k.replace("relative z-score", "rel z").replace("z-score", "z").replace(" | ", "\n").replace("beta-neutral", "β-neut.").replace("BIZD hedge", "BIZD") for k in keys], fontsize=6.5)
ax[1].set_ylabel("Sharpe after costs"); ax[1].legend(fontsize=7); ax[1].set_title("(b) Sharpe by version")
save("n2_neutral_versions")

json.dump(OUT, open(FIG / "neutral_numbers.json", "w"), indent=1, default=float)
rows = []
for k, r in RES.items():
    e = r["exposure"]
    rows.append({"version": k, "IS_sr": r["IS"]["sr"], "IS_t": r["IS"]["t"], "OOS_sr": r["OOS"]["sr"], "OOS_t": r["OOS"]["t"],
                 "OOSx20_sr": r["OOS_ex2020"]["sr"], "last12m%": r["last12m"], "bL": r["leg_beta"]["long"], "bS": r["leg_beta"]["short"],
                 "beta_all": e["all"]["beta_sector"], "t_b": e["all"]["t_sector"], "beta_IS": e["IS"]["beta_sector"], "beta_OOS": e["OOS"]["beta_sector"],
                 "t_bOOS": e["OOS"]["t_sector"], "g_baa": e["all"]["gamma_baa"], "t_g": e["all"]["t_baa"], "a_IS_t": e["IS"]["t_alpha"],
                 "a_OOS_t": e["OOS"]["t_alpha"], "wk1": r["sector_weeks"][1], "wk5": r["sector_weeks"][5]})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("full regression, write-up version, all:", RES[orig]["exposure"]["all"]["full"])
print("final reversal attribution:", OUT["final_reversal"])
