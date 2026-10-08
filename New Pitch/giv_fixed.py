"""
Corrected GIV estimation, plus the tests that decide whether the pitch survives.

WHAT WENT WRONG THE FIRST TIME
------------------------------
Gabaix-Koijen's demand equation is

    dq_it = alpha_i - zeta * dp_t + lambda_i' eta_t + u_it

and the GIV works because subtracting the cross-sectional mean kills BOTH the
common factor AND the -zeta*dp_t term -- the latter only because, with investor
sectors as units, dp_t is ONE aggregate return, identical across units.

With stocks as units the price term is dp_it, the stock's OWN return, which is
not common and does not demean away. The residual therefore still contains each
stock's demand response to its own price move. Measured on the full sample:
mean corr(u_it, idiosyncratic same-day return) = +0.43, positive for 100% of
names. The instrument was contaminated by exactly the reverse causality it was
supposed to purge, which is why naive OLS (0.00302) and "GIV" (0.00304) agreed
and why the implied multiplier was ~30 against Gabaix-Koijen's ~5.

THE FIX
-------
Project the stock's own idiosyncratic return out of the flow when extracting
the idiosyncratic shock:

    dq_check_it = lambda_check_i' eta_t + b_i * dp_check_it + u_it

so u_it is orthogonal to the stock's own relative price move by construction.
Aggregating u into Z = sum_i S_i u_it then gives a demand shock that is not
mechanically a function of the aggregate return.

This is a real restriction, not a free lunch: it removes genuine demand
response as well as mechanical chasing, so it biases the estimate DOWNWARD.
Read the corrected M as a lower bound and the naive number as an upper bound.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

import giv as G


# --------------------------------------------------------------------------
# corrected shock extraction
# --------------------------------------------------------------------------

def extract_shocks_controlling_price(dq, ret, n_factors=3):
    """Residualise flow on common factors AND the stock's own relative return.

    Returns (eta, u, beta) where beta_i is each stock's loading on its own
    idiosyncratic return -- the flow-chases-price coefficient, worth inspecting
    in its own right.
    """
    dq_chk, _ = G.demean_cross_section(dq)
    ret_chk, _ = G.demean_cross_section(ret)

    eta, _, var_ratio, _ = G.extract_factors(dq_chk, n_factors=n_factors)

    cols = [c for c in dq_chk.columns if c in ret_chk.columns]
    u = pd.DataFrame(index=dq_chk.index, columns=cols, dtype=float)
    betas = {}

    E = eta.reindex(dq_chk.index)
    for c in cols:
        d = pd.concat([dq_chk[c].rename("y"), ret_chk[c].rename("p"), E], axis=1).dropna()
        if len(d) < 30:
            continue
        X = sm.add_constant(d[["p"] + list(E.columns)])
        fit = sm.OLS(d.y, X).fit()
        u.loc[d.index, c] = fit.resid
        betas[c] = fit.params["p"]

    return eta, u.dropna(axis=1, how="all"), pd.Series(betas, name="beta_price"), var_ratio


def contamination_check(u, ret):
    """How much same-day return response survives in the shock. Want ~0."""
    ret_chk, _ = G.demean_cross_section(ret)
    cols = [c for c in u.columns if c in ret_chk.columns]
    c = pd.Series({k: u[k].corr(ret_chk[k]) for k in cols}).dropna()
    return {"mean_corr": c.mean(), "median_corr": c.median(),
            "share_positive": (c > 0).mean(), "n": len(c)}


# --------------------------------------------------------------------------
# resampling to lower frequency
# --------------------------------------------------------------------------

def resample_panel(dq, ret, weights, freq="W"):
    """Aggregate to weekly/monthly.

    Flows sum over the period; returns compound (log returns add); weights take
    the period-end value. Gabaix-Koijen work quarterly precisely because flows
    and prices clear over weeks, not hours -- so frequency is a first-order
    robustness dimension, not a footnote.
    """
    dq_r = dq.resample(freq).sum(min_count=1)
    ret_r = ret.resample(freq).sum(min_count=1)
    w_r = weights.resample(freq).last()
    return dq_r, ret_r, w_r


def index_return_from_wide(ret, weights):
    w = weights.reindex_like(ret).where(ret.notna())
    w = w.div(w.sum(axis=1), axis=0)
    return (ret * w).sum(axis=1, min_count=1)


# --------------------------------------------------------------------------
# the headline estimate
# --------------------------------------------------------------------------

def run_giv(dq, ret, weights, n_factors=3, control_price=True, lag_Z=0):
    """One full GIV estimate. Returns a dict of everything worth reporting."""
    if control_price:
        eta, u, beta, var_ratio = extract_shocks_controlling_price(dq, ret, n_factors)
    else:
        dq_chk, _ = G.demean_cross_section(dq)
        eta, _, var_ratio, u = G.extract_factors(dq_chk, n_factors=n_factors)
        beta = pd.Series(dtype=float)

    cols = [c for c in u.columns if c in dq.columns]
    z = G.build_giv(dq[cols], weights[cols], u=u[cols])
    dp = index_return_from_wide(ret, weights)

    Z = z.Z.shift(lag_Z) if lag_Z else z.Z

    fit = G.estimate_multiplier(dp, Z, controls=eta)
    naive = G.naive_ols(dp, z.dq_S)
    contam = contamination_check(u, ret)

    M = fit.params["Z"]
    return {
        "M": M,
        "t": fit.tvalues["Z"],
        "n": int(fit.nobs),
        "naive": naive.params["dq_S"],
        "naive_t": naive.tvalues["dq_S"],
        # Units. dq is in BASIS POINTS of shares outstanding, which at a given
        # price is basis points of market cap. dp is a LOG RETURN (1.0 = 100%).
        # So a flow of 1% of cap = 100bp implies a log return of 100*M, which
        # as a percentage is 100*M*100 = M*1e4.
        # Gabaix-Koijen's "$1 in raises value by $5" is 5% per 1% of cap.
        "pct_move_per_1pct_flow": M * 1e4,
        "corr_Z_rawflow": z.Z.corr(z.dq_S),
        "contam_corr": contam["mean_corr"],
        "idio_share": 1 - var_ratio.head(n_factors).sum(),
        "mean_beta_price": beta.mean() if len(beta) else np.nan,
        "_z": z, "_eta": eta, "_dp": dp, "_u": u, "_fit": fit,
    }


def summarise(rows, title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    df = pd.DataFrame(rows)
    show = [c for c in ["label", "M", "t", "pct_move_per_1pct_flow", "naive",
                        "corr_Z_rawflow", "contam_corr", "idio_share", "n"]
            if c in df.columns]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    return df
