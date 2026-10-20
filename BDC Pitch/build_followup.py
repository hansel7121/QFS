"""Follow-up tests after the review of the Private Credit NAV write-up. Run after build_report_figs.py,
build_residual_test.py, build_residual_conditional.py, build_reversal_check.py and collect_benchmark_prices.py.

  F1  heatmap of the training-period Sharpe grid WITHOUT the x2 display multiplier (true values, with HAC t-stats)
  F2  is the out-of-sample profit just reversal? Regress the z-score strategy's daily returns on a plain reversal
      strategy (same holding period, same buckets, sorted on past 20-day return vs peers)
  F3  short-hold re-optimization: z-score vs reversal-free z-score vs plain reversal, after costs; parameters picked on
      2013-2019 only, then 2020-2026 out of sample
  F4  is reversal special in BDCs? Weekly reversal rank correlation in BDCs vs mortgage REITs vs regional banks (2012-2019)
Outputs: report_figs/g1_..g4_*.png, report_figs/followup_numbers.json.
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
BLUE, ORANGE, GREY, GREEN = "#2f5d8a", "#d17a2b", "#888888", "#4f8a4f"
OUT = {}


def hac_t(x, lags):
    x = pd.Series(x).dropna()
    return sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).tvalues.iloc[0]


def save(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------- data (identical to build_report_figs.py)
H = pd.read_csv(DATA / "nav_history.csv", parse_dates=["period", "public"])
H = H[H.clean & (H.ticker != "GECC") & ~((H.ticker == "CSWC") & (H.period < "2016-01-01"))].sort_values(["ticker", "period"])
p = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
C = p.pivot(index="date", columns="ticker", values="close")
A = p.pivot(index="date", columns="ticker", values="adj_close")
V = p.pivot(index="date", columns="ticker", values="volume")
dates = C.index
bdcs = [t for t in H.ticker.unique() if t in C]
R = np.log(A[bdcs]).diff()
rel = R.sub(R.mean(1), axis=0)
rr = R.fillna(0)

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

# reversal-free z-score: each day, cross-sectional residual of Z on past 5/20/60-day return vs peers
Zres = pd.DataFrame(np.nan, index=dates, columns=bdcs)
Zv, P5, P20, P60 = Z.values, past[5].values, past[20].values, past[60].values
for i in range(len(dates)):
    ok = ~(np.isnan(Zv[i]) | np.isnan(P5[i]) | np.isnan(P20[i]) | np.isnan(P60[i]))
    if ok.sum() < 10: continue
    X = np.column_stack([np.ones(ok.sum()), P5[i, ok], P20[i, ok], P60[i, ok]])
    beta = np.linalg.lstsq(X, Zv[i, ok], rcond=None)[0]
    Zres.values[i, ok] = Zv[i, ok] - X @ beta
REV = past[20]                                                    # plain reversal: long recent losers vs peers

# cost model (identical to build_report_figs.py): tiered per-side cost by 20-day median dollar volume + 1%/yr borrow
DV = (C[bdcs] * V[bdcs]).rolling(20).median().shift(1)
TIERS = [(2e7, 5.0), (5e6, 10.0), (1e6, 20.0)]; TIER_LOW = 40.0
COST = pd.DataFrame(np.select([DV >= a for a, _ in TIERS], [b for _, b in TIERS], TIER_LOW), index=dates, columns=bdcs).where(DV.notna())
BORROW = 0.01 / 252


def strat(sig, H_, q, step, start="2013-03-01"):
    """Every `step` days rank BDCs by `sig`, trade at the next day's close, hold H_ days in overlapping tranches
    (each 1/(H_/step) of capital). Long the lowest q, short the highest q. Returns net, gross daily returns."""
    wl = np.zeros((len(dates), len(bdcs))); ws = np.zeros_like(wl)
    col = {t: i for i, t in enumerate(bdcs)}
    for i in range(4, len(dates) - 2, step):
        s = sig.iloc[i].where(DV.iloc[i].notna()).dropna()
        if len(s) < 9: continue
        lo = [col[x] for x in s[s <= s.quantile(q)].index]; hi_ = [col[x] for x in s[s >= s.quantile(1 - q)].index]
        wl[i + 2:i + 2 + H_, lo] += 1 / len(lo) / (H_ / step); ws[i + 2:i + 2 + H_, hi_] += 1 / len(hi_) / (H_ / step)
    WL, WS = pd.DataFrame(wl, index=dates, columns=bdcs), pd.DataFrame(ws, index=dates, columns=bdcs)
    cost = ((WL.diff().abs() + WS.diff().abs()) * COST.fillna(TIER_LOW)).sum(1) / 1e4 + WS.sum(1) * BORROW
    gross = (WL * rr).sum(1) - (WS * rr).sum(1)
    on = (WL.sum(1) > 0) & (dates >= start)
    return (gross - cost)[on], gross[on]


def stats_(x):
    return {"ann": float(x.mean() * 252 * 100), "sr": float(x.mean() / x.std() * np.sqrt(252)), "t": float(hac_t(x, 10))}


# ================================================================ F1: true (unscaled) heatmap
NUMS = json.load(open(FIG / "numbers.json"))
grid = pd.DataFrame(NUMS["opt"]["grid"]); grid_t = pd.DataFrame(NUMS["opt"]["grid_t"])
grid.index = grid.index.astype(int); grid_t.index = grid_t.index.astype(int)
grid = grid.sort_index(); grid_t = grid_t.sort_index()
fig, ax = plt.subplots(figsize=(6.4, 5.0))
gv = grid.values.astype(float)
im = ax.imshow(gv, cmap="YlGn", vmin=0, vmax=1.2, aspect="auto")
for i in range(gv.shape[0]):
    for j in range(gv.shape[1]):
        ax.text(j, i, f"{gv[i, j]:.2f}\n(t {grid_t.values[i, j]:.1f})", ha="center", va="center", fontsize=7,
                color="black", fontweight="bold" if (grid.index[i] == NUMS["opt"]["H"] and abs(float(grid.columns[j]) - NUMS["opt"]["q"]) < 1e-3) else None)
ax.set_xticks(range(gv.shape[1])); ax.set_xticklabels(["cheapest/richest 20%", "25%", "33%", "40%", "50%"], fontsize=7)
ax.set_yticks(range(gv.shape[0])); ax.set_yticklabels([f"hold {h} days" for h in grid.index], fontsize=7)
ax.set_title("Training-period (2013–2019) Sharpe after costs — true values, no rescaling\n(HAC t-stat in brackets; bold = chosen cell)", fontsize=8)
ax.grid(False); plt.colorbar(im, ax=ax, fraction=0.04)
save("g1_heatmap_true")
OUT["F1"] = {"best_H": NUMS["opt"]["H"], "best_q": NUMS["opt"]["q"], "best_sr": NUMS["opt"]["sr_is"], "best_t": NUMS["opt"]["t_is"],
             "max_t_in_grid": float(grid_t.values.max()), "bt": NUMS["bt"], "bt_ex2020": NUMS["bt_ex2020"]}

# ================================================================ F2: is the profit just reversal?
bH, bq = NUMS["opt"]["H"], NUMS["opt"]["q"]
z_net, z_gross = strat(Z, bH, bq, 3)
r_net, r_gross = strat(REV, bH, bq, 3)
F2 = {}
for era, m in [("IS", z_gross.index < SPLIT), ("OOS", z_gross.index >= SPLIT)]:
    y, x = z_gross[m], r_gross.reindex(z_gross.index)[m]
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    F2[era] = {"alpha_ann": float(fit.params.iloc[0] * 252 * 100), "t_alpha": float(fit.tvalues.iloc[0]), "beta": float(fit.params.iloc[1]),
               "r2": float(fit.rsquared), "corr": float(y.corr(x)), "z_gross": stats_(y), "rev_gross": stats_(x),
               "z_net": stats_(z_net[m]), "rev_net": stats_(r_net.reindex(z_net.index)[m])}
# F2b: control for reversal at every horizon at once (5, 20, 60, 250 days) and compare turnover
rev_k = {k: strat(rel.rolling(k).sum(), bH, bq, 3)[1] for k in (5, 20, 60, 250)}
Xall = pd.DataFrame(rev_k).reindex(z_gross.index)
for era, m in [("IS", z_gross.index < SPLIT), ("OOS", z_gross.index >= SPLIT)]:
    ok = m & Xall.notna().all(1).values
    fit = sm.OLS(z_gross[ok], sm.add_constant(Xall[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    F2[era]["multi"] = {"alpha_ann": float(fit.params.iloc[0] * 252 * 100), "t_alpha": float(fit.tvalues.iloc[0]), "r2": float(fit.rsquared),
                        "betas": {str(k): float(fit.params[k]) for k in rev_k}, "t_betas": {str(k): float(fit.tvalues[k]) for k in rev_k}}


def turnover(sig, H_, q, step):
    n, g_ = strat(sig, H_, q, step)
    return float(((g_ - n).mean() - 0) * 252 * 100)                # annual cost drag (%), proxy for turnover x cost


fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
for lab, s_, c in [("z-score strategy (gross)", z_gross, BLUE), ("plain reversal strategy (gross)", r_gross, GREY)]:
    ax[0].plot(np.exp(s_.cumsum()), color=c, label=lab)
ax[0].axvline(SPLIT, c="r", ls="--", lw=1); ax[0].set_ylabel("growth of $1, before costs"); ax[0].legend(fontsize=7, loc="upper left")
ax[0].set_title(f"(a) z-score vs plain 20d reversal\nsame rules: hold {bH}d, {bq:.0%} buckets")
b = [F2["IS"]["z_gross"]["ann"], F2["IS"]["alpha_ann"], F2["IS"]["multi"]["alpha_ann"],
     F2["OOS"]["z_gross"]["ann"], F2["OOS"]["alpha_ann"], F2["OOS"]["multi"]["alpha_ann"]]
ts = [F2["IS"]["z_gross"]["t"], F2["IS"]["t_alpha"], F2["IS"]["multi"]["t_alpha"],
      F2["OOS"]["z_gross"]["t"], F2["OOS"]["t_alpha"], F2["OOS"]["multi"]["t_alpha"]]
xs = [0, 1, 2, 4, 5, 6]
ax[1].bar(xs, b, color=[BLUE, ORANGE, "#8c4a14"] * 2)
for xx, v, t in zip(xs, b, ts):
    ax[1].text(xx, v + 0.4, f"t={t:.1f}", ha="center", fontsize=7)
ax[1].set_xticks(xs); ax[1].set_xticklabels(["total", "minus 20d\nreversal", "minus all\nhorizons"] * 2, fontsize=7)
ax[1].text(1, -4.2, "2013–2019 (training)", ha="center", fontsize=8); ax[1].text(5, -4.2, "2020–2026 (out of sample)", ha="center", fontsize=8)
ax[1].axhline(0, c="k", lw=0.8); ax[1].set_ylabel("% per year, before costs"); ax[1].set_ylim(-5, max(b) * 1.15)
ax[1].set_title("(b) z-score strategy return:\ntotal vs part reversal portfolios cannot explain")
save("g2_reversal_attribution")
F2["cost_drag_ann"] = {"z-score": turnover(Z, bH, bq, 3), "reversal 20d": turnover(REV, bH, bq, 3)}
OUT["F2"] = F2

# ================================================================ F3: short-hold re-optimization
Hs, qs = [3, 5, 8, 12, 20], [0.25, 1 / 3, 0.5]
SIGS = {"z-score": Z, "reversal-free z-score": Zres, "plain reversal (past 20d)": REV}
F3 = {"grid": [], "best": {}}
for lab, sig in SIGS.items():
    best = None
    for H_ in Hs:
        for q in qs:
            net, gross = strat(sig, H_, q, 1)
            ism = net.index < SPLIT
            row = {"signal": lab, "H": H_, "q": q, "is_net": stats_(net[ism]), "is_gross": stats_(gross[ism]),
                   "oos_net": stats_(net[~ism]), "oos_gross": stats_(gross[~ism])}
            F3["grid"].append(row)
            if best is None or row["is_net"]["sr"] > best["is_net"]["sr"]: best = row; best_series = net
    F3["best"][lab] = best; F3["best"][lab]["_series"] = best_series
# figure: gross vs net IS Sharpe by holding period (q = 1/3) for each signal, plus OOS equity of each best
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))
for (lab, c) in zip(SIGS, [BLUE, GREEN, GREY]):
    rows = [r for r in F3["grid"] if r["signal"] == lab and abs(r["q"] - 1 / 3) < 1e-6]
    ax[0].plot([r["H"] for r in rows], [r["is_gross"]["sr"] for r in rows], color=c, ls="--", marker="o", ms=3)
    ax[0].plot([r["H"] for r in rows], [r["is_net"]["sr"] for r in rows], color=c, marker="o", ms=3, label=lab)
ax[0].axhline(0, c="k", lw=0.8); ax[0].set_xlabel("holding period (days)"); ax[0].set_ylabel("Sharpe, 2013–2019")
ax[0].set_title("(a) Training Sharpe by hold (33% buckets)\ndashed = before costs, solid = after costs"); ax[0].legend(fontsize=7)
for (lab, c) in zip(SIGS, [BLUE, GREEN, GREY]):
    s_ = F3["best"][lab]["_series"]; bst = F3["best"][lab]
    ax[1].plot(np.exp(s_.cumsum()), color=c, label=f"{lab}: hold {bst['H']}d, {bst['q']:.0%}")
ax[1].axvline(SPLIT, c="r", ls="--", lw=1); ax[1].axhline(1, c="k", lw=0.8)
ax[1].set_ylabel("growth of $1, after costs"); ax[1].legend(fontsize=7, loc="upper left")
ax[1].set_title("(b) Best training setting per signal; right of red line = out of sample")
save("g3_short_hold")
for lab in SIGS: F3["best"][lab].pop("_series")
OUT["F3"] = F3

# ================================================================ F4: is reversal special in BDCs?
bp = pd.read_csv(DATA / "benchmark_prices.csv", parse_dates=["date"])
groups = {"BDCs": A[bdcs]}
for grp, d in bp.groupby("group"):
    groups[grp] = d.pivot(index="date", columns="ticker", values="adj_close").reindex(dates)
wk = dates[::5]
F4 = []; ICS = {}
for grp, px in groups.items():
    Rg = np.log(px).diff(); relg = Rg.sub(Rg.mean(1), axis=0)
    p20 = relg.rolling(20).sum()
    for H_ in (5, 20):
        fwd = sum(relg.shift(-k) for k in range(2, H_ + 2))
        ix = wk[np.searchsorted(dates, wk) + H_ + 1 <= last_tr]
        ic = pd.Series({d: p20.loc[d].corr(fwd.loc[d], method="spearman") for d in ix
                        if p20.loc[d].notna().sum() >= 10 and fwd.loc[d].notna().sum() >= 10})
        ICS[(grp, H_)] = ic
        F4.append({"group": grp, "H": H_, "ic": float(ic.mean()), "t": float(hac_t(ic, max(1, H_ // 5))),
                   "n_names": float(p20.loc[ix].notna().sum(1).mean())})
for grp in groups:
    if grp == "BDCs": continue
    for H_ in (5, 20):
        dif = (ICS[("BDCs", H_)] - ICS[(grp, H_)]).dropna()
        F4.append({"group": f"BDCs minus {grp}", "H": H_, "ic": float(dif.mean()), "t": float(hac_t(dif, max(1, H_ // 5))), "n_names": np.nan})
F4 = pd.DataFrame(F4)
fig, ax = plt.subplots(figsize=(6.4, 3.3))
gl = list(groups); k = np.arange(len(gl))
for i, (H_, c) in enumerate([(5, "#b3cde3"), (20, BLUE)]):
    z = F4[F4.H == H_].set_index("group").reindex(gl)
    ax.bar(k + (i - 0.5) * 0.38, z.ic, 0.38, color=c, label=f"next {H_} days")
    for xx, v, t in zip(k + (i - 0.5) * 0.38, z.ic, z.t):
        ax.text(xx, v - 0.004, f"t={t:.1f}", ha="center", va="top", fontsize=7)
ax.axhline(0, c="k", lw=0.8); ax.set_xticks(k); ax.set_xticklabels(gl); ax.set_ylim(F4[F4.group.isin(gl)].ic.min() * 1.2, 0.005)
ax.set_ylabel("avg weekly rank correlation:\npast 20d return vs future return")
ax.set_title("Short-term reversal by group, 2012–2019 (more negative = stronger reversal)"); ax.legend(fontsize=7, loc="lower right")
save("g4_reversal_groups")
OUT["F4"] = F4.to_dict("records")

json.dump(OUT, open(FIG / "followup_numbers.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print("F1", OUT["F1"]["best_sr"], OUT["F1"]["best_t"], "max t", OUT["F1"]["max_t_in_grid"])
print("F2", json.dumps(F2, indent=1))
for lab in SIGS:
    b = F3["best"][lab]
    print("F3", lab, b["H"], round(b["q"], 2), "IS net", {k: round(v, 2) for k, v in b["is_net"].items()}, "IS gross sr", round(b["is_gross"]["sr"], 2),
          "OOS net", {k: round(v, 2) for k, v in b["oos_net"].items()})
g = pd.DataFrame([{"signal": r["signal"], "H": r["H"], "q": round(r["q"], 2), "is_gross_sr": r["is_gross"]["sr"], "is_net_sr": r["is_net"]["sr"],
                   "is_net_t": r["is_net"]["t"], "oos_net_sr": r["oos_net"]["sr"]} for r in F3["grid"]])
print(g.round(2).to_string(index=False))
print(F4.round(3).to_string(index=False))
