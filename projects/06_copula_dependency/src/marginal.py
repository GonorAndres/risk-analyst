"""Marginal filtering via GARCH and probability integral transform.

Fits GARCH(1,1) to each asset's return series, extracts standardised
residuals, and applies the probability integral transform (PIT) to
produce pseudo-uniform marginals suitable for copula estimation.

References:
    - Joe (2014), Ch. 10: inference functions for margins (IFM).
    - McNeil, Frey & Embrechts (2015), Ch. 7.2: GARCH-filtered copula.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from scipy import stats

from risk_analyst.models.copula import pit_transform
from risk_analyst.models.volatility import fit_garch


def filter_marginals(
    returns: pd.DataFrame,
    garch_dist: str = "t",
) -> tuple[np.ndarray, list[ARCHModelResult]]:
    """Fit GARCH(1,1) per asset and apply PIT to standardised residuals.

    For each column in *returns*:
        1. Fit GARCH(1,1) with the specified innovation distribution.
        2. Extract standardised residuals z_t = epsilon_t / sigma_t.
        3. Apply the empirical PIT: rank/(n+1).

    Parameters
    ----------
    returns : pd.DataFrame
        Asset log-returns, one column per asset.
    garch_dist : str
        Innovation distribution for GARCH: ``"normal"``, ``"t"``, or ``"skewt"``.

    Returns
    -------
    tuple[np.ndarray, list[ARCHModelResult]]
        - ``u_data``: pseudo-uniform observations of shape (n, d), each
          column in (0, 1).
        - ``garch_models``: list of fitted GARCH model results (one per asset).
    """
    n_assets = returns.shape[1]
    garch_models: list[ARCHModelResult] = []
    std_resid_list: list[np.ndarray] = []

    for j in range(n_assets):
        col = returns.iloc[:, j]
        fitted = fit_garch(col.values, p=1, q=1, dist=garch_dist)
        garch_models.append(fitted)

        # Extract standardised residuals (drop NaN from warm-up)
        raw = fitted.std_resid
        if isinstance(raw, pd.Series):
            resid = raw.dropna().values
        else:
            resid = np.asarray(raw, dtype=np.float64)
            resid = resid[~np.isnan(resid)]
        std_resid_list.append(resid)

    # Align lengths (all should be the same after GARCH warm-up)
    min_len = min(len(r) for r in std_resid_list)
    aligned = np.column_stack([r[-min_len:] for r in std_resid_list])

    # Apply empirical PIT column-by-column
    u_data = pit_transform(aligned, method="empirical")

    return u_data, garch_models


def inverse_pit(
    u_data: np.ndarray,
    garch_models: list[ARCHModelResult],
    forecast_sigmas: np.ndarray,
) -> np.ndarray:
    """Invert the PIT + GARCH transform to produce return scenarios.

    For each asset j:
        1. Map uniform u_j to standardised residual z_j via the inverse
           of the fitted innovation CDF.
        2. Multiply by the forecast conditional volatility:
           r_j = sigma_j * z_j  (assuming zero mean).

    Parameters
    ----------
    u_data : np.ndarray
        Copula samples in (0, 1) of shape (n_samples, d).
    garch_models : list[ARCHModelResult]
        One fitted GARCH model per asset (for the innovation distribution).
    forecast_sigmas : np.ndarray
        1-step-ahead conditional standard deviations, shape (d,), in
        decimal form (e.g. 0.01 = 1 %).

    Returns
    -------
    np.ndarray
        Simulated return scenarios, shape (n_samples, d).
    """
    n_samples, d = u_data.shape
    u_clipped = np.clip(u_data, 1e-10, 1 - 1e-10)
    returns_sim = np.empty_like(u_data)

    for j in range(d):
        model = garch_models[j]
        params = model.params
        dist_name = model.model.distribution.name

        # Inverse CDF of the innovation distribution
        if dist_name == "Normal":
            z = stats.norm.ppf(u_clipped[:, j])
        elif dist_name == "Standardized Student's t":
            nu = params["nu"]
            z = stats.t.ppf(u_clipped[:, j], df=nu)
        else:
            # Fallback: use empirical quantiles from standardised residuals
            raw = model.std_resid
            if isinstance(raw, pd.Series):
                resid = raw.dropna().values
            else:
                resid = np.asarray(raw, dtype=np.float64)
                resid = resid[~np.isnan(resid)]
            z = np.interp(
                u_clipped[:, j],
                np.linspace(0, 1, len(resid)),
                np.sort(resid),
            )

        returns_sim[:, j] = forecast_sigmas[j] * z

    return returns_sim
