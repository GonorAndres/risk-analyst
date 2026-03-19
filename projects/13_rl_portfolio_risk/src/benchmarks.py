"""Benchmark portfolio allocation strategies.

Implements static strategies for comparison against the RL agent:
equal weight (1/N), mean-variance, and risk parity.

References:
- DeMiguel et al. (2009), "Optimal versus naive diversification" (1/N benchmark).
- Markowitz (1952), "Portfolio selection" (mean-variance).
- Maillard, Roncalli & Teiletche (2010), "The properties of equally weighted
  risk contribution portfolios" (risk parity).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from trainer import compute_cvar


def equal_weight(n_assets: int) -> NDArray[np.float64]:
    """Equal weight (1/N) allocation.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Returns
    -------
    weights : ndarray of shape (n_assets,)
        Uniform weights summing to 1.
    """
    return np.ones(n_assets, dtype=np.float64) / n_assets


def mean_variance(
    returns: NDArray[np.float64],
    risk_aversion: float = 1.0,
) -> NDArray[np.float64]:
    """Mean-variance optimal weights (long-only).

    w* = Sigma^{-1} mu  (unconstrained), then clip to [0, 1] and normalise.

    Uses pseudo-inverse for numerical stability when the covariance
    matrix is near-singular.

    Parameters
    ----------
    returns : ndarray of shape (T, n_assets)
        Historical return matrix.
    risk_aversion : float
        Risk aversion parameter (higher = more conservative).

    Returns
    -------
    weights : ndarray of shape (n_assets,)
        Normalised non-negative weights.
    """
    mu = np.mean(returns, axis=0)
    sigma = np.cov(returns, rowvar=False)

    # Pseudo-inverse for stability
    sigma_inv = np.linalg.pinv(sigma)
    raw_weights = sigma_inv @ mu / risk_aversion

    # Long-only: clip negatives, then normalise
    raw_weights = np.clip(raw_weights, 0.0, None)
    total = np.sum(raw_weights)
    if total < 1e-12:
        # Fallback to equal weight if all weights are zero
        return np.ones(len(mu), dtype=np.float64) / len(mu)
    return (raw_weights / total).astype(np.float64)


def risk_parity(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Risk parity (inverse volatility) allocation.

    w_i = (1 / sigma_i) / sum(1 / sigma_j)

    Parameters
    ----------
    returns : ndarray of shape (T, n_assets)
        Historical return matrix.

    Returns
    -------
    weights : ndarray of shape (n_assets,)
        Inverse-volatility weights summing to 1.
    """
    sigmas = np.std(returns, axis=0, ddof=1)
    # Guard against zero volatility
    sigmas = np.maximum(sigmas, 1e-12)
    inv_vol = 1.0 / sigmas
    return (inv_vol / np.sum(inv_vol)).astype(np.float64)


def run_benchmark(
    strategy_weights: NDArray[np.float64],
    returns_data: NDArray[np.float64],
    cost_rate: float = 0.001,
) -> dict:
    """Backtest a static portfolio strategy.

    Parameters
    ----------
    strategy_weights : ndarray of shape (n_assets,)
        Fixed portfolio weights.
    returns_data : ndarray of shape (T, n_assets)
        Return matrix for backtesting.
    cost_rate : float
        Proportional transaction cost rate (applied at start).

    Returns
    -------
    results : dict
        Keys: total_return, sharpe, cvar_95, max_drawdown,
              cumulative_returns, drawdown, portfolio_returns.
    """
    # Portfolio returns (static weights, no rebalancing cost except initial)
    portfolio_returns = returns_data @ strategy_weights

    # Initial cost for establishing position
    initial_cost = cost_rate * np.sum(np.abs(strategy_weights))
    portfolio_returns_adj = portfolio_returns.copy()
    portfolio_returns_adj[0] -= initial_cost

    # Cumulative returns
    cumulative = np.cumprod(1.0 + portfolio_returns_adj)

    # Total return
    total_return = float(cumulative[-1] - 1.0)

    # Sharpe ratio (annualised)
    if np.std(portfolio_returns_adj) > 1e-12:
        sharpe = float(
            np.mean(portfolio_returns_adj)
            / np.std(portfolio_returns_adj)
            * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    # CVaR at 95%
    cvar_95 = compute_cvar(portfolio_returns_adj, alpha=0.95)

    # Drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / np.maximum(peak, 1e-12)
    max_drawdown = float(np.max(drawdown))

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown,
        "cumulative_returns": cumulative,
        "drawdown": drawdown,
        "portfolio_returns": portfolio_returns_adj,
    }
