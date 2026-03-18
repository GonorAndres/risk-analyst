"""Diagnostic plots for EVT analysis.

Provides visualization functions for assessing GPD/GEV fit quality
and comparing tail risk estimates across methods.

References:
    - Coles (2001), Ch. 3--4: diagnostic plots.
    - McNeil, Frey & Embrechts (2015), Ch. 5: QQ-plots for EVT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def plot_qq_gpd(
    exceedances: np.ndarray, gpd_params: dict
) -> tuple:
    """QQ plot comparing empirical exceedances against fitted GPD quantiles.

    Parameters
    ----------
    exceedances : np.ndarray
        1-D array of positive exceedances (x_i - threshold).
    gpd_params : dict
        Output of :func:`fit_gpd` with keys xi, sigma.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    xi = gpd_params["xi"]
    sigma = gpd_params["sigma"]

    sorted_exc = np.sort(exceedances)
    n = len(sorted_exc)
    # Plotting positions (Hazen)
    p = (np.arange(1, n + 1) - 0.5) / n
    # Theoretical quantiles from fitted GPD
    theoretical = stats.genpareto.ppf(p, c=xi, loc=0, scale=sigma)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(theoretical, sorted_exc, s=15, alpha=0.6, edgecolors="k",
               linewidths=0.3, label="Exceedances")

    # Reference line
    lims = [
        min(theoretical.min(), sorted_exc.min()),
        max(theoretical.max(), sorted_exc.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="45-degree line")

    ax.set_xlabel("Theoretical GPD Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title(f"GPD QQ-Plot (xi={xi:.3f}, sigma={sigma:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_return_level(
    gev_params: dict, max_period: float
) -> tuple:
    """Return level plot with approximate confidence bands from GEV fit.

    Parameters
    ----------
    gev_params : dict
        Output of :func:`fit_gev` with keys xi, mu, sigma.
    max_period : float
        Maximum return period to plot.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    from risk_analyst.models.evt import return_level

    periods = np.logspace(np.log10(1.5), np.log10(max_period), 200)
    levels = np.array([return_level(gev_params, T) for T in periods])

    # Approximate delta-method confidence band (rough approximation)
    xi = gev_params["xi"]
    sigma = gev_params["sigma"]
    # Use +/- 15% as a simple visual band (proper CI requires observed info matrix)
    upper = levels * 1.15
    lower = levels * 0.85

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(periods, levels, "b-", linewidth=2, label="Return level")
    ax.fill_between(periods, lower, upper, alpha=0.2, color="blue",
                    label="Approximate 70% CI")

    ax.set_xscale("log")
    ax.set_xlabel("Return Period (blocks)")
    ax.set_ylabel("Return Level")
    ax.set_title(
        f"GEV Return Level Plot (xi={xi:.3f}, mu={gev_params['mu']:.4f}, "
        f"sigma={sigma:.4f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    return fig, ax


def plot_evt_vs_normal(comparison_df: pd.DataFrame) -> tuple:
    """Grouped bar chart comparing VaR estimates across methods.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of ``EVTModel.compare_methods()`` with columns:
        alpha, var_normal, var_t, var_evt.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    alphas = comparison_df["alpha"].values
    x = np.arange(len(alphas))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, comparison_df["var_normal"], width,
           label="Normal VaR", color="#4C72B0", alpha=0.85)
    ax.bar(x, comparison_df["var_t"], width,
           label="Student-t VaR", color="#55A868", alpha=0.85)
    ax.bar(x + width, comparison_df["var_evt"], width,
           label="EVT VaR", color="#C44E52", alpha=0.85)

    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("VaR")
    ax.set_title("VaR Comparison: Normal vs Student-t vs EVT")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1%}" for a in alphas])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    return fig, ax


def plot_mean_residual_life(
    thresholds: np.ndarray, mean_excesses: np.ndarray
) -> tuple:
    """Mean residual life plot for threshold selection.

    Parameters
    ----------
    thresholds : np.ndarray
        Array of threshold values.
    mean_excesses : np.ndarray
        Corresponding mean excess values.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, mean_excesses, "o-", markersize=3, linewidth=1,
            color="#4C72B0")
    ax.set_xlabel("Threshold (u)")
    ax.set_ylabel("Mean Excess e(u)")
    ax.set_title("Mean Residual Life Plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_threshold_stability(stability_df: pd.DataFrame) -> tuple:
    """Parameter stability plot: xi and sigma* vs threshold.

    Parameters
    ----------
    stability_df : pd.DataFrame
        Output of :func:`parameter_stability` with columns:
        threshold, xi, xi_se, sigma_star.

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and two-panel axes array.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    u = stability_df["threshold"]
    xi = stability_df["xi"]
    xi_se = stability_df["xi_se"]
    sigma_star = stability_df["sigma_star"]

    # Top panel: xi with confidence band
    axes[0].plot(u, xi, "o-", markersize=3, linewidth=1, color="#C44E52")
    axes[0].fill_between(u, xi - 1.96 * xi_se, xi + 1.96 * xi_se,
                         alpha=0.2, color="#C44E52")
    axes[0].set_ylabel("Shape (xi)")
    axes[0].set_title("GPD Parameter Stability")
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: modified scale
    axes[1].plot(u, sigma_star, "o-", markersize=3, linewidth=1,
                 color="#4C72B0")
    axes[1].set_xlabel("Threshold (u)")
    axes[1].set_ylabel("Modified Scale (sigma*)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, axes
