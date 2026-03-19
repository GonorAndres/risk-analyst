"""CVA computation engine.

Computes unilateral CVA, DVA, and bilateral CVA from exposure profiles
and credit parameters.

Reference: Gregory (2020), Ch. 12 -- CVA.
"""

from __future__ import annotations

import numpy as np
from credit import (
    default_probability,
)
from exposure import apply_netting
from numpy.typing import NDArray


def compute_cva(
    ee: NDArray[np.float64],
    times: NDArray[np.float64],
    hazard_rate: float,
    recovery: float,
) -> float:
    """Compute unilateral Credit Valuation Adjustment.

    CVA = (1 - R) * sum_i EE(t_i) * [PD(t_i) - PD(t_{i-1})] * D(t_i)

    where D(t) = exp(-r_f * t) is the discount factor.  For simplicity,
    we use risk-free rate = 0 (i.e., D(t) = 1) since the exposure
    already incorporates discounting through the swap valuation.

    Parameters
    ----------
    ee : ndarray of shape (n_times,)
        Expected Exposure profile.
    times : ndarray of shape (n_times,)
        Time grid.
    hazard_rate : float
        Counterparty hazard rate (annualised).
    recovery : float
        Counterparty recovery rate.

    Returns
    -------
    float
        CVA value.
    """
    lgd = 1.0 - recovery
    cva = 0.0

    for i in range(1, len(times)):
        # Marginal default probability in [t_{i-1}, t_i]
        pd_prev = default_probability(hazard_rate, times[i - 1])
        pd_curr = default_probability(hazard_rate, times[i])
        marginal_pd = pd_curr - pd_prev

        # Discount factor (using risk-free rate approximation)
        discount = np.exp(-0.03 * times[i])  # simple flat discount

        cva += lgd * ee[i] * marginal_pd * discount

    return float(cva)


def compute_dva(
    ene: NDArray[np.float64],
    times: NDArray[np.float64],
    own_hazard: float,
    recovery: float,
) -> float:
    """Compute Debit Valuation Adjustment (own default benefit).

    DVA = (1 - R_own) * sum_i |ENE(t_i)| * [PD_own(t_i) - PD_own(t_{i-1})] * D(t_i)

    DVA is symmetric to CVA but uses own default probability and
    negative exposure.

    Parameters
    ----------
    ene : ndarray of shape (n_times,)
        Expected Negative Exposure profile (negative values).
    times : ndarray of shape (n_times,)
        Time grid.
    own_hazard : float
        Own hazard rate (annualised).
    recovery : float
        Own recovery rate.

    Returns
    -------
    float
        DVA value (positive means benefit from own default).
    """
    lgd = 1.0 - recovery
    dva = 0.0

    for i in range(1, len(times)):
        pd_prev = default_probability(own_hazard, times[i - 1])
        pd_curr = default_probability(own_hazard, times[i])
        marginal_pd = pd_curr - pd_prev

        discount = np.exp(-0.03 * times[i])

        # ENE is negative; |ENE| = -ENE
        dva += lgd * np.abs(ene[i]) * marginal_pd * discount

    return float(dva)


def compute_bilateral_cva(
    ee: NDArray[np.float64],
    ene: NDArray[np.float64],
    times: NDArray[np.float64],
    cpty_hazard: float,
    own_hazard: float,
    recovery: float,
) -> dict:
    """Compute bilateral CVA (CVA, DVA, and BCVA).

    BCVA = CVA - DVA

    Parameters
    ----------
    ee : ndarray of shape (n_times,)
        Expected Exposure profile.
    ene : ndarray of shape (n_times,)
        Expected Negative Exposure profile.
    times : ndarray of shape (n_times,)
        Time grid.
    cpty_hazard : float
        Counterparty hazard rate.
    own_hazard : float
        Own hazard rate.
    recovery : float
        Recovery rate (same for both parties for simplicity).

    Returns
    -------
    dict
        Dictionary with keys 'cva', 'dva', 'bcva'.
    """
    cva = compute_cva(ee, times, cpty_hazard, recovery)
    dva = compute_dva(ene, times, own_hazard, recovery)
    bcva = cva - dva

    return {"cva": cva, "dva": dva, "bcva": bcva}


def cva_by_netting_set(
    trades_mtm: list[NDArray[np.float64]],
    times: NDArray[np.float64],
    hazard_rate: float,
    recovery: float,
) -> dict:
    """Compare CVA with and without netting.

    Parameters
    ----------
    trades_mtm : list of ndarray
        MTM values for each trade, each shape (n_paths, n_times).
    times : ndarray of shape (n_times,)
        Time grid.
    hazard_rate : float
        Counterparty hazard rate.
    recovery : float
        Counterparty recovery rate.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'gross_cva': CVA without netting (sum of individual CVAs)
        - 'net_cva': CVA with netting
        - 'netting_benefit': gross_cva - net_cva
        - 'benefit_pct': percentage reduction
    """
    # Gross CVA: sum of individual trade CVAs
    gross_cva = 0.0
    for mtm in trades_mtm:
        # Individual trade exposure
        positive_exp = np.maximum(mtm, 0.0)
        ee_i = np.mean(positive_exp, axis=0)
        gross_cva += compute_cva(ee_i, times, hazard_rate, recovery)

    # Net CVA: exposure after netting
    netted_exposure = apply_netting(trades_mtm)
    ee_net = np.mean(netted_exposure, axis=0)
    net_cva = compute_cva(ee_net, times, hazard_rate, recovery)

    netting_benefit = gross_cva - net_cva
    benefit_pct = (netting_benefit / gross_cva * 100.0) if gross_cva > 0 else 0.0

    return {
        "gross_cva": gross_cva,
        "net_cva": net_cva,
        "netting_benefit": netting_benefit,
        "benefit_pct": benefit_pct,
    }
