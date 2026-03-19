"""GARCH-family volatility model fitting and forecasting.

Wraps the ``arch`` library to provide a clean interface for fitting
symmetric (GARCH) and asymmetric (GJR-GARCH, EGARCH) conditional
heteroscedasticity models, producing multi-step volatility forecasts,
and computing conditional risk measures (VaR, ES).

The ``arch`` library works with *percentage* returns internally
(returns * 100).  All public functions in this module accept returns
in *decimal* form (e.g., 0.01 for 1 %) and handle the scaling
transparently.

References:
    - Bollerslev (1986): GARCH(p,q).
    - Glosten, Jagannathan & Runkle (1993): GJR-GARCH.
    - Nelson (1991): EGARCH.
    - McNeil, Frey & Embrechts (2015), Ch. 4: conditional volatility models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from scipy import stats

# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

_DIST_MAP: dict[str, str] = {
    "normal": "normal",
    "t": "t",
    "skewt": "skewt",
}


def _validate_dist(dist: str) -> str:
    """Map a user-friendly distribution name to the ``arch`` convention."""
    key = dist.lower()
    if key not in _DIST_MAP:
        raise ValueError(
            f"Unknown distribution '{dist}'. Choose from {list(_DIST_MAP.keys())}."
        )
    return _DIST_MAP[key]


def fit_garch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
) -> ARCHModelResult:
    """Fit a GARCH(p, q) model.

    sigma_t^2 = omega + sum_{i=1}^{q} alpha_i * epsilon_{t-i}^2
                      + sum_{j=1}^{p} beta_j * sigma_{t-j}^2

    Parameters
    ----------
    returns : array-like
        Decimal returns (e.g. 0.01 for 1 %).
    p : int
        Number of lagged conditional variance terms.
    q : int
        Number of lagged squared return terms.
    dist : str
        Innovation distribution: ``"normal"``, ``"t"``, or ``"skewt"``.

    Returns
    -------
    ARCHModelResult
        Fitted model result.
    """
    pct_returns = np.asarray(returns, dtype=np.float64) * 100.0
    model = arch_model(
        pct_returns,
        vol="Garch",
        p=p,
        q=q,
        dist=_validate_dist(dist),
        rescale=False,
    )
    result: ARCHModelResult = model.fit(disp="off")
    return result


def fit_gjr_garch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
) -> ARCHModelResult:
    """Fit a GJR-GARCH(p, o, q) model (asymmetric / leverage effect).

    sigma_t^2 = omega + sum alpha_i * epsilon_{t-i}^2
                      + sum gamma_j * epsilon_{t-j}^2 * I(epsilon_{t-j}<0)
                      + sum beta_k * sigma_{t-k}^2

    Parameters
    ----------
    returns : array-like
        Decimal returns.
    p, o, q : int
        Lag orders for variance, asymmetry, and ARCH terms respectively.
    dist : str
        Innovation distribution.

    Returns
    -------
    ARCHModelResult
    """
    pct_returns = np.asarray(returns, dtype=np.float64) * 100.0
    model = arch_model(
        pct_returns,
        vol="Garch",
        p=p,
        o=o,
        q=q,
        dist=_validate_dist(dist),
        rescale=False,
    )
    result: ARCHModelResult = model.fit(disp="off")
    return result


def fit_egarch(
    returns: np.ndarray | pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "t",
) -> ARCHModelResult:
    """Fit an EGARCH(p, o, q) model.

    ln(sigma_t^2) = omega + sum alpha_i * |z_{t-i}|
                          + sum gamma_j * z_{t-j}
                          + sum beta_k * ln(sigma_{t-k}^2)

    where z_t = epsilon_t / sigma_t.  The log specification guarantees
    sigma_t^2 > 0 without parameter constraints.

    Parameters
    ----------
    returns : array-like
        Decimal returns.
    p, o, q : int
        Lag orders.
    dist : str
        Innovation distribution.

    Returns
    -------
    ARCHModelResult
    """
    pct_returns = np.asarray(returns, dtype=np.float64) * 100.0
    model = arch_model(
        pct_returns,
        vol="EGARCH",
        p=p,
        o=o,
        q=q,
        dist=_validate_dist(dist),
        rescale=False,
    )
    result: ARCHModelResult = model.fit(disp="off")
    return result


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def forecast_volatility(
    fitted_model: ARCHModelResult,
    horizon: int = 1,
) -> pd.DataFrame:
    """Produce h-step-ahead conditional variance forecasts.

    Returns annualized *decimal* volatility (sigma, not sigma^2) by
    converting back from the percentage scale used internally by ``arch``.

    Parameters
    ----------
    fitted_model : ARCHModelResult
        A previously fitted GARCH-family model.
    horizon : int
        Number of steps ahead.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``h1``, ``h2``, ... containing the
        forecasted conditional standard deviation at each horizon
        (in decimal form, i.e. 0.01 = 1 %).
    """
    forecast = fitted_model.forecast(horizon=horizon)
    # forecast.variance is in pct^2; convert to decimal sigma
    variance_pct = forecast.variance.dropna()
    sigma_decimal = np.sqrt(variance_pct) / 100.0
    return sigma_decimal


# ---------------------------------------------------------------------------
# Conditional risk measures
# ---------------------------------------------------------------------------


def _std_resid_array(fitted_model: ARCHModelResult) -> np.ndarray:
    """Extract standardized residuals as a clean numpy array (no NaN)."""
    raw = fitted_model.std_resid
    if isinstance(raw, pd.Series):
        return raw.dropna().values
    arr = np.asarray(raw, dtype=np.float64)
    return arr[~np.isnan(arr)]


def _last_conditional_sigma(fitted_model: ARCHModelResult) -> float:
    """Extract the latest 1-step-ahead conditional sigma in decimal form."""
    fcast = fitted_model.forecast(horizon=1)
    last_var_pct = float(fcast.variance.dropna().iloc[-1, 0])
    return np.sqrt(last_var_pct) / 100.0


def _innovation_quantile(fitted_model: ARCHModelResult, alpha: float) -> float:
    """Return the alpha-quantile of the fitted innovation distribution.

    ``alpha`` is the VaR confidence level (e.g. 0.99), so the left-tail
    quantile is at probability ``1 - alpha``.
    """
    params = fitted_model.params
    dist_name = fitted_model.model.distribution.name

    if dist_name == "Normal":
        return float(stats.norm.ppf(1 - alpha))
    elif dist_name == "Standardized Student's t":
        nu = params["nu"]
        return float(stats.t.ppf(1 - alpha, df=nu))
    elif dist_name.startswith("Standardized Skew"):
        # skewed-t: fall back to empirical std-residual quantile
        std_resid = fitted_model.std_resid.dropna()
        return float(np.quantile(std_resid, 1 - alpha))
    else:
        # Fallback: empirical quantile of standardized residuals
        std_resid = fitted_model.std_resid.dropna()
        return float(np.quantile(std_resid, 1 - alpha))


def conditional_var(
    fitted_model: ARCHModelResult,
    alpha: float,
) -> float:
    """Conditional Value-at-Risk using the fitted GARCH model.

    VaR_{alpha} = -(mu + sigma_t * q_{1-alpha})

    where q_{1-alpha} is the quantile of the standardized innovation
    distribution and sigma_t is the latest conditional standard deviation.

    Parameters
    ----------
    fitted_model : ARCHModelResult
        Fitted GARCH-family model.
    alpha : float
        Confidence level (e.g. 0.95 or 0.99).

    Returns
    -------
    float
        Conditional VaR expressed as a positive loss (decimal returns).
    """
    sigma = _last_conditional_sigma(fitted_model)
    q = _innovation_quantile(fitted_model, alpha)
    mu = fitted_model.params.get("mu", 0.0) / 100.0  # pct -> decimal
    return float(-(mu + sigma * q))


def conditional_es(
    fitted_model: ARCHModelResult,
    alpha: float,
) -> float:
    """Conditional Expected Shortfall using the fitted GARCH model.

    ES_{alpha} = E[-r_t | r_t <= -VaR_{alpha}]

    Estimated from the standardized residuals rescaled by the current
    conditional volatility.

    Parameters
    ----------
    fitted_model : ARCHModelResult
        Fitted GARCH-family model.
    alpha : float
        Confidence level (e.g. 0.95 or 0.99).

    Returns
    -------
    float
        Conditional ES (>= conditional VaR).
    """
    sigma = _last_conditional_sigma(fitted_model)
    mu = fitted_model.params.get("mu", 0.0) / 100.0

    std_resid = _std_resid_array(fitted_model)
    q = _innovation_quantile(fitted_model, alpha)
    tail = std_resid[std_resid <= q]

    if len(tail) == 0:
        # Not enough data in the tail -- fall back to VaR
        return conditional_var(fitted_model, alpha)

    tail_mean = float(np.mean(tail))
    return float(-(mu + sigma * tail_mean))
