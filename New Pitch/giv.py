"""
Granular IV machinery, following Gabaix & Koijen (NBER w28204, w28967).

The model, at the stock level. For stock i on day t, institutional demand is

    dq_it = alpha_i - zeta * dp_it + lambda_i' eta_t + u_it                 (1)

with dp_it the stock return, zeta the demand elasticity, eta_t common shocks,
u_it idiosyncratic. The identifying assumption is only E[u_it eta_t] = 0.

Write X_E for the equal-weighted and X_S for the size-weighted cross-sectional
average. Because the common term lambda_i' eta_t is (under uniform loadings)
identical across i, the wedge

    Z_t = dq_St - dq_Et = u_St - u_Et                                       (2)

contains idiosyncratic shocks ONLY, so it is orthogonal to eta_t by
construction. That is the instrument. Then

    dp_t = M * Z_t + beta' eta_t + e_t          (OLS, M read off directly)   (3)
    dq_Et = -zeta * dp_t + beta' eta_t + eps_t  (2SLS, dp_t instrumented)    (4)

With non-uniform loadings you first PCA the cross-sectionally demeaned dq to
extract r factors and include them as controls (w28967 eq. 34-35).

Two refinements from the papers that matter in practice:
  * Optimal weights are Gamma' = S'Q with Q = I - L(L'L)^-1 L' the projection
    orthogonal to the loadings -- the vector closest to size weights while
    orthogonal to the factor structure (w28204 Prop. 2).
  * Under heteroskedastic u_it, replace equal weights with inverse-variance
    "quasi-equal" weights Etilde_i proportional to 1/sigma_ui^2.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# --------------------------------------------------------------------------
# factor extraction
# --------------------------------------------------------------------------

def demean_cross_section(dq, weights=None):
    """Subtract the cross-sectional mean each day: dq_check_it = dq_it - dq_Et.

    Removes both the common factor loading (under uniform loadings) and the
    -zeta * dp_t term, which is what leaves idiosyncratic variation behind.
    """
    if weights is None:
        centre = dq.mean(axis=1)
    else:
        w = weights.reindex_like(dq).where(dq.notna())
        w = w.div(w.sum(axis=1), axis=0)
        centre = (dq * w).sum(axis=1, min_count=1)
    return dq.sub(centre, axis=0), centre


def extract_factors(dq_check, n_factors=3, standardise=True):
    """PCA on the demeaned panel -> (factors eta_t, loadings, variance ratios).

    Implements w28967 eq. (34): dq_check_it = lambda_check_i' eta_t + u_check_it.
    Returns the residual u as well; that is what the GIV is built from.
    """
    X = dq_check.dropna(axis=1, thresh=int(0.8 * len(dq_check)))
    X = X.fillna(0.0)
    if standardise:
        sd = X.std().replace(0, np.nan)
        X = X.div(sd, axis=1).fillna(0.0)

    U, S, Vt = np.linalg.svd(X.values, full_matrices=False)
    var_ratio = (S ** 2) / (S ** 2).sum()

    eta = pd.DataFrame(U[:, :n_factors] * S[:n_factors],
                       index=X.index,
                       columns=[f"eta{i+1}" for i in range(n_factors)])
    loadings = pd.DataFrame(Vt[:n_factors].T, index=X.columns, columns=eta.columns)

    fitted = eta.values @ Vt[:n_factors]
    resid = pd.DataFrame(X.values - fitted, index=X.index, columns=X.columns)
    if standardise:
        resid = resid.mul(sd, axis=1)

    return eta, loadings, pd.Series(var_ratio, name="var_ratio"), resid


# --------------------------------------------------------------------------
# the instrument
# --------------------------------------------------------------------------

def inverse_variance_weights(u):
    """Quasi-equal weights Etilde_i ~ 1/sigma_ui^2, normalised to sum to 1."""
    var = u.var()
    inv = 1.0 / var.replace(0, np.nan)
    return inv / inv.sum()


def _weighted_mean(src, weights):
    w = weights.reindex_like(src).where(src.notna())
    w = w.div(w.sum(axis=1), axis=0)
    return (src * w).sum(axis=1, min_count=1)


def build_giv(dq, weights, u=None, quasi_equal=True):
    """Z_t = u_St - u_Et, the size-weighted minus (quasi-)equal-weighted wedge.

    Pass u (the PCA residual) to build the instrument from residualised shocks,
    which is the general-loadings case. Pass nothing for the uniform-loadings
    case, where the wedge alone kills the common term.

    Columns returned:
      Z         the instrument
      u_S, u_E  the two averages Z is the difference of
      dq_S      size-weighted average of the RAW flow, kept separate so the
                biased naive regression has the right right-hand side even when
                the instrument is built from residuals
    """
    src = dq if u is None else u

    size_wtd = _weighted_mean(src, weights)

    if quasi_equal:
        e = inverse_variance_weights(src)
        ew = pd.DataFrame(np.tile(e.values, (len(src), 1)),
                          index=src.index, columns=src.columns)
        ew = ew.where(src.notna())
        ew = ew.div(ew.sum(axis=1), axis=0)
        equal_wtd = (src * ew).sum(axis=1, min_count=1)
    else:
        equal_wtd = src.mean(axis=1)

    return pd.DataFrame({
        "Z": size_wtd - equal_wtd,
        "u_S": size_wtd,
        "u_E": equal_wtd,
        "dq_S": _weighted_mean(dq, weights),
        "dq_E": dq.mean(axis=1),
    })


def optimal_giv_weights(size_weights, loadings):
    """Gamma' = S'Q, Q = I - L(L'L)^-1 L'  (w28204 Prop. 2).

    The vector closest to size weights while orthogonal to the factor loadings,
    which minimises the asymptotic variance of the estimator.
    """
    L = loadings.values
    L = np.column_stack([np.ones(len(L)), L])      # always include the constant factor
    Q = np.eye(len(L)) - L @ np.linalg.pinv(L.T @ L) @ L.T
    S = size_weights.reindex(loadings.index).fillna(0.0).values
    return pd.Series(Q.T @ S, index=loadings.index, name="gamma")


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------

def estimate_multiplier(dp, Z, controls=None, hac_lags=5):
    """OLS of eq. (3): dp_t = M Z_t + beta' eta_t + e_t. Returns M and the fit."""
    X = pd.DataFrame({"Z": Z})
    if controls is not None:
        X = X.join(controls)
    d = pd.concat([dp.rename("dp"), X], axis=1).dropna()
    res = sm.OLS(d.dp, sm.add_constant(d[X.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return res


def estimate_elasticity(dq_E, dp, Z, controls=None, hac_lags=5):
    """2SLS of eq. (4): dq_Et on dp_t instrumented by Z_t. Returns zeta = -coef.

    Reports the first-stage F so instrument strength is visible rather than
    assumed -- with one instrument this is the relevant weak-IV diagnostic.
    """
    from statsmodels.sandbox.regression.gmm import IV2SLS

    cols = {"dp": dp, "dq_E": dq_E, "Z": Z}
    d = pd.DataFrame(cols)
    if controls is not None:
        d = d.join(controls)
    d = d.dropna()

    ctrl = [c for c in d.columns if c.startswith("eta")]
    exog = sm.add_constant(d[["dp"] + ctrl])
    instr = sm.add_constant(d[["Z"] + ctrl])

    fit = IV2SLS(d.dq_E, exog, instr).fit()

    first = sm.OLS(d.dp, sm.add_constant(d[["Z"] + ctrl])).fit()
    f_stat = first.tvalues["Z"] ** 2

    return fit, f_stat


def naive_ols(dp, dq_S, hac_lags=5):
    """The biased benchmark: regress the return on the raw size-weighted flow.

    Both sides load on eta_t, so this is not an elasticity -- it is the number
    the GIV exists to correct. Reported so the correction is visible.
    """
    d = pd.concat([dp.rename("dp"), dq_S.rename("dq_S")], axis=1).dropna()
    return sm.OLS(d.dp, sm.add_constant(d[["dq_S"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
