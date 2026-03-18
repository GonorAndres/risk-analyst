"""Value-at-Risk and Expected Shortfall implementations.

Implements three VaR estimation approaches:
    1. Historical simulation (empirical quantile)
    2. Parametric (variance-covariance under normality)
    3. Monte Carlo (bootstrap resampling)

Plus Expected Shortfall (CVaR) as the conditional tail expectation.

References:
    - Jorion (2007), Ch. 11-12: VaR methods.
    - Acerbi & Tasche (2002): coherence of Expected Shortfall.
    - McNeil, Frey & Embrechts (2015), Ch. 2.2: risk measures.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def historical_var(losses: np.ndarray, alpha: float) -> float:
    """Historical simulation VaR: empirical quantile of the loss distribution.

    VaR_alpha = F_L^{-1}(alpha) = quantile(losses, alpha)

    Parameters
    ----------
    losses : np.ndarray
        1-D array of portfolio losses (positive = loss).
    alpha : float
        Confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        VaR estimate at the *alpha* confidence level.
    """
    return float(np.quantile(losses, alpha))


def parametric_var(losses: np.ndarray, alpha: float) -> float:
    """Parametric (variance-covariance) VaR under the normal assumption.

    VaR_alpha = mu_L + sigma_L * Phi^{-1}(alpha)

    where mu_L and sigma_L are the sample mean and standard deviation of
    losses, and Phi^{-1} is the standard normal quantile function.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of portfolio losses.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        Parametric VaR estimate.
    """
    mu: float = float(np.mean(losses))
    sigma: float = float(np.std(losses, ddof=1))
    z_alpha: float = float(stats.norm.ppf(alpha))
    return mu + sigma * z_alpha


def monte_carlo_var(
    losses: np.ndarray,
    alpha: float,
    n_sims: int = 10_000,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR via bootstrap resampling of the historical loss series.

    Draws *n_sims* samples (with replacement) from the observed loss
    distribution and returns the *alpha*-quantile of the resampled
    distribution.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of portfolio losses.
    alpha : float
        Confidence level.
    n_sims : int
        Number of bootstrap samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Monte Carlo VaR estimate.
    """
    rng = np.random.default_rng(seed)
    simulated = rng.choice(losses, size=n_sims, replace=True)
    return float(np.quantile(simulated, alpha))


def expected_shortfall(losses: np.ndarray, alpha: float) -> float:
    """Expected Shortfall (CVaR): average loss exceeding VaR.

    ES_alpha = E[L | L >= VaR_alpha]
             = (1 / (1 - alpha)) * integral_{alpha}^{1} VaR_u du

    Estimated empirically as the mean of losses at or above the
    alpha-quantile.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of portfolio losses.
    alpha : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    float
        Expected Shortfall estimate.
    """
    var = historical_var(losses, alpha)
    tail_losses = losses[losses >= var]
    if len(tail_losses) == 0:
        return var
    return float(np.mean(tail_losses))
