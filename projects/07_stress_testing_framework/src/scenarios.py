"""Scenario definitions and generators for stress testing.

Provides DFAST-style regulatory scenarios (baseline / adverse / severely adverse),
historical crisis replays (GFC 2008, COVID 2020, SVB 2023), and stochastic
Monte Carlo scenario generation from a multivariate normal model.

References:
    - Federal Reserve Board, DFAST 2023 Supervisory Scenarios.
    - Basel Committee (2018), Stress Testing Principles.
    - Breuer & Csiszar (2013), Systematic stress tests with entropic
      plausibility constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. DFAST regulatory scenarios
# ---------------------------------------------------------------------------

def get_dfast_scenarios() -> dict[str, pd.DataFrame]:
    """Return three synthetic DFAST-like scenario DataFrames.

    Each DataFrame has 9 quarters (Q1--Q9) and columns:
        gdp_growth, unemployment, equity_index, interest_rate_10y,
        credit_spread, house_price_index.

    Values are realistic but synthetic, calibrated to match typical
    DFAST severity gradations.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"baseline"``, ``"adverse"``, ``"severely_adverse"``.
    """
    quarters = [f"Q{i}" for i in range(1, 10)]

    # -- Baseline: moderate expansion ----------------------------------------
    baseline = pd.DataFrame({
        "quarter": quarters,
        "gdp_growth": [0.022, 0.023, 0.021, 0.020, 0.021, 0.022, 0.023, 0.022, 0.021],
        "unemployment": [0.040, 0.039, 0.039, 0.038, 0.038, 0.037, 0.037, 0.036, 0.036],
        "equity_index": [0.02, 0.02, 0.015, 0.018, 0.02, 0.022, 0.018, 0.019, 0.02],
        "interest_rate_10y": [0.040, 0.041, 0.042, 0.042, 0.043, 0.043, 0.044, 0.044, 0.044],
        "credit_spread": [0.015, 0.015, 0.014, 0.014, 0.013, 0.013, 0.013, 0.012, 0.012],
        "house_price_index": [0.010, 0.012, 0.011, 0.010, 0.011, 0.012, 0.013, 0.012, 0.011],
    }).set_index("quarter")

    # -- Adverse: mild recession ---------------------------------------------
    adverse = pd.DataFrame({
        "quarter": quarters,
        "gdp_growth": [-0.010, -0.020, -0.015, -0.005, 0.000, 0.005, 0.010, 0.012, 0.015],
        "unemployment": [0.055, 0.065, 0.075, 0.080, 0.082, 0.080, 0.078, 0.075, 0.070],
        "equity_index": [-0.10, -0.12, -0.08, -0.04, -0.02, 0.00, 0.02, 0.03, 0.04],
        "interest_rate_10y": [0.030, 0.025, 0.020, 0.018, 0.017, 0.018, 0.020, 0.022, 0.025],
        "credit_spread": [0.030, 0.040, 0.045, 0.042, 0.038, 0.035, 0.030, 0.028, 0.025],
        "house_price_index": [-0.02, -0.04, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01],
    }).set_index("quarter")

    # -- Severely Adverse: deep recession ------------------------------------
    severely_adverse = pd.DataFrame({
        "quarter": quarters,
        "gdp_growth": [-0.030, -0.050, -0.040, -0.025, -0.010, 0.000, 0.005, 0.010, 0.012],
        "unemployment": [0.065, 0.080, 0.095, 0.100, 0.100, 0.098, 0.095, 0.090, 0.085],
        "equity_index": [-0.20, -0.25, -0.15, -0.08, -0.04, -0.01, 0.02, 0.04, 0.05],
        "interest_rate_10y": [0.020, 0.012, 0.008, 0.007, 0.008, 0.010, 0.015, 0.018, 0.022],
        "credit_spread": [0.050, 0.070, 0.080, 0.075, 0.065, 0.055, 0.045, 0.038, 0.032],
        "house_price_index": [-0.05, -0.10, -0.12, -0.10, -0.07, -0.04, -0.02, 0.00, 0.01],
    }).set_index("quarter")

    return {
        "baseline": baseline,
        "adverse": adverse,
        "severely_adverse": severely_adverse,
    }


# ---------------------------------------------------------------------------
# 2. Historical crisis scenarios
# ---------------------------------------------------------------------------

def get_historical_scenarios() -> dict[str, dict[str, float]]:
    """Return macro factor shocks for notable historical crises.

    Each scenario is a dict mapping factor names to one-period shock
    magnitudes calibrated to approximate peak observed moves.

    Returns
    -------
    dict[str, dict[str, float]]
        Keys: ``"gfc_2008"``, ``"covid_2020"``, ``"svb_2023"``.
    """
    return {
        "gfc_2008": {
            "gdp_growth": -0.04,
            "unemployment": 0.05,
            "equity_index": -0.50,
            "interest_rate_10y": -0.03,
            "credit_spread": 0.04,
            "house_price_index": -0.20,
        },
        "covid_2020": {
            "gdp_growth": -0.09,
            "unemployment": 0.10,
            "equity_index": -0.34,
            "interest_rate_10y": -0.015,
            "credit_spread": 0.03,
            "house_price_index": -0.02,
        },
        "svb_2023": {
            "gdp_growth": -0.005,
            "unemployment": 0.005,
            "equity_index": -0.10,
            "interest_rate_10y": 0.015,
            "credit_spread": 0.02,
            "house_price_index": -0.03,
        },
    }


# ---------------------------------------------------------------------------
# 3. Stochastic (Monte Carlo) scenario generation
# ---------------------------------------------------------------------------

def generate_stochastic_scenarios(
    n_scenarios: int,
    factor_means: np.ndarray,
    factor_cov: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Generate Monte Carlo scenarios from a multivariate normal distribution.

    Parameters
    ----------
    n_scenarios : int
        Number of scenarios to draw.
    factor_means : np.ndarray
        Mean vector of length *k* (number of factors).
    factor_cov : np.ndarray
        Covariance matrix of shape (*k*, *k*).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (*n_scenarios*, *k*) with simulated factor realisations.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(factor_means, factor_cov, size=n_scenarios)
