"""Portfolio risk measures via Monte Carlo simulation.

Computes Value-at-Risk (VaR) and Expected Shortfall (ES / CVaR) by
simulating future portfolio returns from the empirical distribution
of historical returns.

Reference: McNeil, Frey & Embrechts (2015), Ch. 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mc_portfolio_var(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    alpha: float = 0.95,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> float:
    """Compute portfolio VaR via Monte Carlo simulation.

    Resamples (with replacement) from historical asset returns to build
    simulated portfolio return distributions, then takes the alpha-quantile
    of the loss distribution.

    Parameters
    ----------
    returns : array of shape (n_obs, n_assets)
        Historical asset returns (simple or log).
    weights : array of shape (n_assets,)
        Portfolio weights (should sum to 1).
    alpha : float
        Confidence level (e.g., 0.95 means 95 % VaR).
    n_sims : int
        Number of Monte Carlo scenarios.
    seed : int or None
        Random seed.

    Returns
    -------
    var : float
        Portfolio VaR (positive number representing a loss).
    """
    returns = np.asarray(returns, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)

    n_obs = returns.shape[0]

    # Resample rows (days) with replacement
    idx = rng.integers(0, n_obs, size=n_sims)
    sim_returns = returns[idx]  # (n_sims, n_assets)

    # Portfolio returns
    portfolio_returns = sim_returns @ weights  # (n_sims,)

    # VaR = negative alpha-quantile of returns (loss is positive)
    var = -float(np.percentile(portfolio_returns, 100 * (1 - alpha)))
    return var


def mc_portfolio_es(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    alpha: float = 0.95,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> float:
    """Compute portfolio Expected Shortfall (CVaR) via Monte Carlo.

    ES = E[-R | -R >= VaR_alpha]

    This is the average loss conditional on losses exceeding VaR.

    Parameters
    ----------
    returns : array of shape (n_obs, n_assets)
        Historical asset returns.
    weights : array of shape (n_assets,)
        Portfolio weights.
    alpha : float
        Confidence level.
    n_sims : int
        Number of Monte Carlo scenarios.
    seed : int or None
        Random seed.

    Returns
    -------
    es : float
        Portfolio Expected Shortfall (positive number).
    """
    returns = np.asarray(returns, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)

    n_obs = returns.shape[0]

    # Resample rows with replacement
    idx = rng.integers(0, n_obs, size=n_sims)
    sim_returns = returns[idx]

    # Portfolio returns
    portfolio_returns = sim_returns @ weights

    # Losses (negate returns)
    losses = -portfolio_returns

    # VaR threshold
    var_threshold = np.percentile(losses, 100 * alpha)

    # ES = mean of losses exceeding VaR
    tail_losses = losses[losses >= var_threshold]

    if len(tail_losses) == 0:
        return float(var_threshold)

    return float(np.mean(tail_losses))
