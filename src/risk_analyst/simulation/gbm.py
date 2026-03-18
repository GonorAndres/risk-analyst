"""Geometric Brownian Motion (GBM) simulation.

Implements single-asset and multi-asset correlated GBM using
Euler-Maruyama discretization.  Multi-asset correlation is handled
via Cholesky decomposition of the correlation matrix.

Reference: Glasserman (2003), Ch. 3 -- Generating Sample Paths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate single-asset GBM paths via exact log-normal solution.

    Uses the analytic solution:
        S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    s0 : float
        Initial asset price.
    mu : float
        Annualised drift (expected return).
    sigma : float
        Annualised volatility.
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    paths : ndarray of shape (n_paths, n_steps + 1)
        Simulated price paths.  paths[:, 0] == s0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Draw all increments at once: (n_paths, n_steps)
    z = rng.standard_normal((n_paths, n_steps))

    # Log-increments: exact discretisation of GBM
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    # Build log-price paths and exponentiate
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(log_increments, axis=1)

    return np.exp(log_paths)


def simulate_gbm_correlated(
    s0_vec: NDArray[np.float64],
    mu_vec: NDArray[np.float64],
    sigma_vec: NDArray[np.float64],
    corr_matrix: NDArray[np.float64],
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate multi-asset correlated GBM paths via Cholesky decomposition.

    For *d* correlated assets, we:
      1. Draw independent standard normals Z ~ N(0, I_d).
      2. Compute L = cholesky(corr_matrix) so that L @ L^T = corr_matrix.
      3. Set correlated increments: W = L @ Z.
      4. Evolve each asset using its own (mu_i, sigma_i) with correlated W_i.

    Parameters
    ----------
    s0_vec : array of shape (n_assets,)
        Initial prices for each asset.
    mu_vec : array of shape (n_assets,)
        Annualised drifts.
    sigma_vec : array of shape (n_assets,)
        Annualised volatilities.
    corr_matrix : array of shape (n_assets, n_assets)
        Correlation matrix (must be symmetric positive-definite).
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    paths : ndarray of shape (n_paths, n_steps + 1, n_assets)
        Simulated correlated price paths.
    """
    s0_vec = np.asarray(s0_vec, dtype=np.float64)
    mu_vec = np.asarray(mu_vec, dtype=np.float64)
    sigma_vec = np.asarray(sigma_vec, dtype=np.float64)
    corr_matrix = np.asarray(corr_matrix, dtype=np.float64)

    n_assets = len(s0_vec)
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    # Cholesky factor: L such that L @ L^T = corr_matrix
    L = np.linalg.cholesky(corr_matrix)

    # Independent normals: (n_paths, n_steps, n_assets)
    z_indep = rng.standard_normal((n_paths, n_steps, n_assets))

    # Correlate: multiply each (n_assets,) vector by L^T
    # z_corr[i, t, :] = L @ z_indep[i, t, :]
    z_corr = np.einsum("ij,ntj->nti", L, z_indep)

    # Log-increments per asset: (n_paths, n_steps, n_assets)
    drift = (mu_vec - 0.5 * sigma_vec**2) * dt  # shape (n_assets,)
    diffusion = sigma_vec * np.sqrt(dt)  # shape (n_assets,)
    log_increments = drift[np.newaxis, np.newaxis, :] + diffusion[np.newaxis, np.newaxis, :] * z_corr

    # Build log-price paths
    log_paths = np.zeros((n_paths, n_steps + 1, n_assets))
    log_paths[:, 0, :] = np.log(s0_vec)[np.newaxis, :]
    log_paths[:, 1:, :] = np.log(s0_vec)[np.newaxis, np.newaxis, :] + np.cumsum(
        log_increments, axis=1
    )

    return np.exp(log_paths)
