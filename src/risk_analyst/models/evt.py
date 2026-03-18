"""Extreme Value Theory model fitting and risk measures.

Provides functions for fitting the Generalized Extreme Value (GEV) and
Generalized Pareto Distribution (GPD) distributions via MLE, computing
EVT-based Value-at-Risk and Expected Shortfall, and estimating return
levels from block maxima fits.

Mathematical background:

    GEV distribution (Fisher--Tippett--Gnedenko theorem):
        H_{xi,mu,sigma}(x) = exp(-(1 + xi*(x - mu)/sigma)^{-1/xi})

    GPD (Pickands--Balkema--de Haan theorem):
        G_{xi,sigma}(y) = 1 - (1 + xi*y/sigma)^{-1/xi}   for xi != 0
        G_{0,sigma}(y)   = 1 - exp(-y/sigma)               for xi = 0

References:
    - de Haan & Ferreira (2006), Ch. 1--3: GEV and GPD theory.
    - McNeil & Frey (2000): estimation of tail-related risk measures.
    - Embrechts, Kluppelberg & Mikosch (1997): Modelling Extremal Events.
    - McNeil, Frey & Embrechts (2015), Ch. 5: EVT for risk management.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fit_gev(data: np.ndarray) -> dict:
    """Fit GEV distribution via scipy.stats.genextreme MLE.

    The GEV encompasses three sub-families:
        - Frechet (xi > 0): heavy-tailed
        - Gumbel  (xi = 0): light-tailed
        - Weibull (xi < 0): bounded upper tail

    Parameters
    ----------
    data : np.ndarray
        1-D array of block maxima observations.

    Returns
    -------
    dict
        Keys: xi (shape), mu (location), sigma (scale), nll (negative
        log-likelihood).

    Notes
    -----
    scipy.stats.genextreme parameterises shape as ``c = -xi`` relative to
    the standard EVT convention, so we negate the fitted ``c``.
    """
    # scipy convention: c = -xi (sign flip)
    c, loc, scale = stats.genextreme.fit(data)
    xi = -c  # convert to EVT convention

    nll = -np.sum(stats.genextreme.logpdf(data, c, loc=loc, scale=scale))

    return {
        "xi": float(xi),
        "mu": float(loc),
        "sigma": float(scale),
        "nll": float(nll),
    }


def fit_gpd(exceedances: np.ndarray, threshold: float) -> dict:
    """Fit GPD via scipy.stats.genpareto MLE.

    Parameters
    ----------
    exceedances : np.ndarray
        1-D array of positive exceedances above *threshold*
        (i.e., x_i - threshold for x_i > threshold).
    threshold : float
        The threshold value u.

    Returns
    -------
    dict
        Keys: xi (shape), sigma (scale), threshold, n_exceed.

    Raises
    ------
    ValueError
        If exceedances contain non-positive values.
    """
    if np.any(exceedances <= 0):
        raise ValueError("Exceedances must be strictly positive.")

    # scipy.stats.genpareto: c = xi, loc fixed at 0 for exceedances
    c, _loc, scale = stats.genpareto.fit(exceedances, floc=0)

    return {
        "xi": float(c),
        "sigma": float(scale),
        "threshold": float(threshold),
        "n_exceed": int(len(exceedances)),
    }


def evt_var(gpd_params: dict, n_total: int, alpha: float) -> float:
    """EVT-based VaR from a fitted GPD (peaks-over-threshold).

    VaR_alpha = u + (sigma / xi) * ((n / N_u * (1 - alpha))^{-xi} - 1)

    where u is the threshold, sigma and xi are GPD parameters, n is
    the total sample size, and N_u is the number of exceedances.

    Parameters
    ----------
    gpd_params : dict
        Output of :func:`fit_gpd` with keys xi, sigma, threshold, n_exceed.
    n_total : int
        Total number of observations in the original sample.
    alpha : float
        Confidence level (e.g. 0.99).

    Returns
    -------
    float
        EVT-based VaR estimate.
    """
    xi = gpd_params["xi"]
    sigma = gpd_params["sigma"]
    u = gpd_params["threshold"]
    n_u = gpd_params["n_exceed"]

    if xi == 0:
        # Exponential tail case
        return u + sigma * np.log(n_total / (n_u * (1 - alpha)))

    return u + (sigma / xi) * ((n_total / (n_u * (1 - alpha))) ** xi - 1)


def evt_es(gpd_params: dict, n_total: int, alpha: float) -> float:
    """EVT-based Expected Shortfall from a fitted GPD.

    ES_alpha = VaR_alpha / (1 - xi) + (sigma - xi * u) / (1 - xi)

    Parameters
    ----------
    gpd_params : dict
        Output of :func:`fit_gpd`.
    n_total : int
        Total number of observations.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        EVT-based ES estimate.

    Raises
    ------
    ValueError
        If xi >= 1 (ES is infinite for xi >= 1).
    """
    xi = gpd_params["xi"]
    sigma = gpd_params["sigma"]
    u = gpd_params["threshold"]

    if xi >= 1:
        raise ValueError(
            f"ES is infinite for xi >= 1 (got xi={xi:.4f})."
        )

    var = evt_var(gpd_params, n_total, alpha)
    return var / (1 - xi) + (sigma - xi * u) / (1 - xi)


def return_level(gev_params: dict, return_period: float) -> float:
    """T-year return level from a fitted GEV distribution.

    z_T = mu + (sigma / xi) * ((-log(1 - 1/T))^{-xi} - 1)   for xi != 0
    z_T = mu - sigma * log(-log(1 - 1/T))                      for xi = 0

    Parameters
    ----------
    gev_params : dict
        Output of :func:`fit_gev` with keys xi, mu, sigma.
    return_period : float
        Return period T (e.g. 10, 50, 100 years/blocks).

    Returns
    -------
    float
        The T-period return level.
    """
    xi = gev_params["xi"]
    mu = gev_params["mu"]
    sigma = gev_params["sigma"]

    y_p = -np.log(1 - 1 / return_period)

    if abs(xi) < 1e-10:
        # Gumbel limit
        return mu - sigma * np.log(y_p)

    return mu + (sigma / xi) * (y_p ** (-xi) - 1)
