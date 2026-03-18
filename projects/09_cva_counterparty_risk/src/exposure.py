"""Exposure profile computation for counterparty credit risk.

Computes Expected Exposure (EE), Potential Future Exposure (PFE),
Expected Negative Exposure (ENE), and Effective Expected Positive
Exposure (EPE) from simulated mark-to-market values.

Reference: Gregory (2020), Ch. 7 -- Quantifying Credit Exposure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_exposure_profiles(
    mtm_values: NDArray[np.float64],
    times: NDArray[np.float64],
) -> dict:
    """Compute exposure profiles from simulated MTM values.

    Parameters
    ----------
    mtm_values : ndarray of shape (n_paths, n_times)
        Mark-to-market values for each path at each time.
    times : ndarray of shape (n_times,)
        Time grid.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ee': Expected Exposure at each time, shape (n_times,)
        - 'pfe_975': 97.5th percentile of positive exposure, shape (n_times,)
        - 'pfe_99': 99th percentile of positive exposure, shape (n_times,)
        - 'ene': Expected Negative Exposure at each time, shape (n_times,)
        - 'epe': scalar, time-weighted average of EE
    """
    # Positive exposure: max(V, 0)
    positive_exposure = np.maximum(mtm_values, 0.0)

    # Expected Exposure: E[max(V, 0)] at each time
    ee = np.mean(positive_exposure, axis=0)

    # Potential Future Exposure: quantiles of positive exposure
    pfe_975 = np.percentile(positive_exposure, 97.5, axis=0)
    pfe_99 = np.percentile(positive_exposure, 99.0, axis=0)

    # Expected Negative Exposure: E[min(V, 0)] at each time (for DVA)
    ene = np.mean(np.minimum(mtm_values, 0.0), axis=0)

    # Effective Expected Positive Exposure: time-weighted average of EE
    # EPE = (1/T) * integral_0^T EE(t) dt  (trapezoidal rule)
    if len(times) > 1:
        epe = float(np.trapezoid(ee, times) / (times[-1] - times[0]))
    else:
        epe = float(ee[0])

    return {
        "ee": ee,
        "pfe_975": pfe_975,
        "pfe_99": pfe_99,
        "ene": ene,
        "epe": epe,
    }


def apply_netting(
    mtm_values_list: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Net multiple trade MTM values and compute exposure.

    V_net = sum of individual V_i
    Exposure = max(V_net, 0)

    Parameters
    ----------
    mtm_values_list : list of ndarray
        Each element has shape (n_paths, n_times) representing MTM
        values of an individual trade.

    Returns
    -------
    ndarray of shape (n_paths, n_times)
        Netted positive exposure: max(sum(V_i), 0).
    """
    # Sum all trade MTM values
    v_net = np.sum(np.array(mtm_values_list), axis=0)
    return np.maximum(v_net, 0.0)


def apply_collateral(
    mtm_values: NDArray[np.float64],
    threshold: float,
    mta: float,
) -> NDArray[np.float64]:
    """Apply collateral to reduce exposure.

    Simplified collateral model:
        collateral = max(V - threshold, 0)
        collateralized_exposure = max(V - collateral + mta, 0)
                                = max(min(V, threshold) + mta, 0)

    Parameters
    ----------
    mtm_values : ndarray of shape (n_paths, n_times)
        Mark-to-market values (or positive exposures).
    threshold : float
        Collateral threshold (amount of exposure before collateral
        is called).
    mta : float
        Minimum transfer amount.

    Returns
    -------
    ndarray of shape (n_paths, n_times)
        Collateralized exposure.
    """
    # Collateral posted = max(V - threshold, 0)
    collateral = np.maximum(mtm_values - threshold, 0.0)
    # Collateralized exposure = max(V - collateral + mta, 0)
    collateralized = np.maximum(mtm_values - collateral + mta, 0.0)
    return collateralized
