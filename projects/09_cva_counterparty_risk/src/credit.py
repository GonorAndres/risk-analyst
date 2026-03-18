"""Default probability and hazard rate modeling.

Implements hazard rate bootstrapping from CDS spreads and survival
probability calculations for counterparty credit risk.

Reference: Gregory (2020), Ch. 10 -- Default Probability, Credit
Spreads, and Funding Costs.
"""

from __future__ import annotations

import numpy as np


def hazard_rate_from_cds(cds_spread: float, recovery: float) -> float:
    """Compute constant hazard rate implied by a CDS spread.

    lambda = s / (1 - R)

    Parameters
    ----------
    cds_spread : float
        CDS par spread (annualised, e.g. 0.01 for 100 bps).
    recovery : float
        Recovery rate (e.g. 0.40).

    Returns
    -------
    float
        Implied hazard rate (annualised).
    """
    return cds_spread / (1.0 - recovery)


def survival_probability(hazard_rate: float, t: float) -> float:
    """Compute survival probability up to time t.

    Q(t) = exp(-lambda * t)

    Parameters
    ----------
    hazard_rate : float
        Constant hazard rate (annualised).
    t : float
        Time horizon in years.

    Returns
    -------
    float
        Survival probability Q(t).
    """
    return float(np.exp(-hazard_rate * t))


def default_probability(hazard_rate: float, t: float) -> float:
    """Compute cumulative default probability up to time t.

    PD(t) = 1 - Q(t) = 1 - exp(-lambda * t)

    Parameters
    ----------
    hazard_rate : float
        Constant hazard rate (annualised).
    t : float
        Time horizon in years.

    Returns
    -------
    float
        Cumulative default probability PD(t).
    """
    return 1.0 - survival_probability(hazard_rate, t)


def marginal_default_prob(hazard_rate: float, t1: float, t2: float) -> float:
    """Compute marginal default probability in the interval [t1, t2].

    P(default in [t1, t2]) = Q(t1) - Q(t2)

    Parameters
    ----------
    hazard_rate : float
        Constant hazard rate (annualised).
    t1 : float
        Start of interval in years.
    t2 : float
        End of interval in years.

    Returns
    -------
    float
        Marginal default probability.
    """
    return survival_probability(hazard_rate, t1) - survival_probability(hazard_rate, t2)


def bootstrap_hazard_rates(
    cds_spreads: dict[float, float],
    recovery: float,
) -> dict[float, float]:
    """Bootstrap piecewise-constant hazard rates from CDS term structure.

    Given CDS par spreads at various tenors, compute the cumulative
    hazard rate at each tenor using a piecewise-constant intensity model.

    For each tenor T_k with spread s_k, we solve for the hazard rate
    lambda_k such that the CDS premium leg equals the protection leg.

    In the simple approximation with flat hazard rate per tenor bucket:
        lambda_k = s_k / (1 - R)

    For the piecewise bootstrap, we use the fact that the cumulative
    hazard at each tenor can be inferred from the market spread.

    Parameters
    ----------
    cds_spreads : dict[float, float]
        Mapping of tenor (years) -> CDS par spread (annualised).
    recovery : float
        Recovery rate.

    Returns
    -------
    dict[float, float]
        Mapping of tenor -> cumulative hazard rate (i.e., the average
        hazard rate that gives the correct survival probability at
        that tenor).
    """
    sorted_tenors = sorted(cds_spreads.keys())
    hazard_rates: dict[float, float] = {}

    prev_tenor = 0.0
    cumulative_hazard = 0.0

    for tenor in sorted_tenors:
        spread = cds_spreads[tenor]
        # Marginal hazard rate for this bucket
        marginal_lambda = spread / (1.0 - recovery)
        dt = tenor - prev_tenor

        # Update cumulative hazard: integral of lambda from 0 to tenor
        cumulative_hazard += marginal_lambda * dt

        # Average hazard rate up to this tenor
        avg_hazard = cumulative_hazard / tenor
        hazard_rates[tenor] = avg_hazard

        prev_tenor = tenor

    return hazard_rates
