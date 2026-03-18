"""Threshold selection diagnostics for peaks-over-threshold (POT) analysis.

Provides tools for choosing an appropriate GPD threshold:
    1. Mean Residual Life (MRL) plot -- linearity above the correct threshold
    2. Parameter stability plot -- GPD xi and sigma* vs threshold
    3. Hill estimator -- non-parametric tail index estimation
    4. Automatic selection via percentile or other heuristics

References:
    - Coles (2001), Ch. 4.3: threshold selection.
    - de Haan & Ferreira (2006), Ch. 4: Hill estimator.
    - McNeil, Frey & Embrechts (2015), Ch. 5.2: POT diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_residual_life(
    losses: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean excess function for a range of thresholds.

    For each threshold u, the mean excess is:
        e(u) = E[X - u | X > u] = (1/N_u) * sum_{x_i > u} (x_i - u)

    If GPD holds above u_0, then e(u) is approximately linear in u
    for u >= u_0.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of loss observations (positive values).
    thresholds : np.ndarray
        1-D array of threshold candidates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (thresholds, mean_excesses) -- arrays of the same length.
        Thresholds with no exceedances are excluded.
    """
    valid_thresholds = []
    mean_excesses = []

    for u in thresholds:
        exceedances = losses[losses > u] - u
        if len(exceedances) > 0:
            valid_thresholds.append(u)
            mean_excesses.append(float(np.mean(exceedances)))

    return np.array(valid_thresholds), np.array(mean_excesses)


def parameter_stability(
    losses: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    """Fit GPD at each threshold and return parameter estimates.

    A good threshold yields stable xi and sigma* = sigma - xi*u
    (the modified scale) as the threshold increases.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of loss observations.
    thresholds : np.ndarray
        1-D array of threshold candidates.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, xi, xi_se, sigma_star.
        Rows where fitting fails or too few exceedances are excluded.
    """
    records = []

    for u in thresholds:
        exceedances = losses[losses > u] - u
        if len(exceedances) < 10:
            # Need enough exceedances for a reliable fit
            continue

        try:
            # scipy.stats.genpareto: c = xi, loc fixed at 0
            c, _loc, scale = stats.genpareto.fit(exceedances, floc=0)
            xi = float(c)
            sigma = float(scale)

            # Approximate standard error of xi via observed Fisher information
            # For GPD with large n_exceed, se(xi) ~ (1 + xi) / sqrt(n)
            n = len(exceedances)
            xi_se = (1 + xi) / np.sqrt(2 * n) if n > 0 else np.nan

            sigma_star = sigma - xi * u

            records.append({
                "threshold": float(u),
                "xi": xi,
                "xi_se": xi_se,
                "sigma_star": sigma_star,
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def hill_estimator(
    losses: np.ndarray, k_range: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hill estimator of the tail index for different numbers of order statistics.

    The Hill estimator for the tail index alpha (where xi = 1/alpha) is:
        H_{k,n} = (1/k) * sum_{i=1}^{k} log(X_{(n-i+1)} / X_{(n-k)})

    where X_{(1)} <= ... <= X_{(n)} are the order statistics.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of positive loss observations.
    k_range : np.ndarray
        1-D array of integer k values (number of upper order statistics).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (k_values, hill_estimates) -- the Hill estimate of xi = 1/alpha
        for each valid k value.
    """
    sorted_losses = np.sort(losses)[::-1]  # descending order
    n = len(sorted_losses)

    valid_k = []
    hill_estimates = []

    for k in k_range:
        k = int(k)
        if k < 2 or k >= n:
            continue

        # X_{(n-k)} in ascending order = sorted_losses[k] in descending order
        log_ratios = np.log(sorted_losses[:k] / sorted_losses[k])
        hill_est = float(np.mean(log_ratios))

        valid_k.append(k)
        hill_estimates.append(hill_est)

    return np.array(valid_k), np.array(hill_estimates)


def select_threshold_auto(
    losses: np.ndarray,
    method: str = "percentile",
    percentile: float = 95.0,
) -> float:
    """Automatic threshold selection.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of loss observations.
    method : str
        Selection method. Currently supported:
        - "percentile": use the given percentile of the loss distribution.
    percentile : float
        Percentile value (0-100) when method="percentile".

    Returns
    -------
    float
        Selected threshold value.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method == "percentile":
        return float(np.percentile(losses, percentile))

    raise ValueError(
        f"Unknown threshold selection method '{method}'. "
        f"Supported: 'percentile'."
    )
