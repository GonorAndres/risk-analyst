"""Diagnostic plots for CVA counterparty risk analysis.

All functions return (fig, ax) tuples for composability and testing.

Reference: Gregory (2020), various figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def plot_exposure_profiles(
    profiles: dict,
    times: NDArray[np.float64],
) -> tuple:
    """Plot Expected Exposure and PFE profiles over time.

    Parameters
    ----------
    profiles : dict
        Output of compute_exposure_profiles with keys 'ee', 'pfe_975',
        'pfe_99'.
    times : ndarray of shape (n_times,)
        Time grid.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, profiles["ee"], linewidth=2, label="EE", color="steelblue")
    ax.plot(
        times,
        profiles["pfe_975"],
        linewidth=2,
        linestyle="--",
        label="PFE (97.5%)",
        color="darkorange",
    )
    ax.plot(
        times,
        profiles["pfe_99"],
        linewidth=2,
        linestyle=":",
        label="PFE (99%)",
        color="firebrick",
    )

    ax.fill_between(times, 0, profiles["ee"], alpha=0.15, color="steelblue")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure")
    ax.set_title("Exposure Profiles: EE and PFE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_cva_waterfall(cva_components: dict) -> tuple:
    """Plot CVA, DVA, BCVA as a waterfall chart.

    Parameters
    ----------
    cva_components : dict
        Dictionary with keys 'cva', 'dva', 'bcva'.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["CVA", "DVA", "BCVA"]
    values = [
        cva_components["cva"],
        -cva_components["dva"],  # DVA is a benefit (negative)
        cva_components["bcva"],
    ]
    colors = ["firebrick", "forestgreen", "steelblue"]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,.0f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Adjustment Value")
    ax.set_title("CVA Waterfall: CVA, DVA, and Bilateral CVA")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    return fig, ax


def plot_netting_benefit(netting_df: pd.DataFrame) -> tuple:
    """Plot gross vs net CVA as a bar chart.

    Parameters
    ----------
    netting_df : pd.DataFrame
        DataFrame with columns 'metric' and 'value', as returned
        by CVAModel.netting_analysis().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    gross = netting_df.loc[netting_df["metric"] == "Gross CVA", "value"].values[0]
    net = netting_df.loc[netting_df["metric"] == "Net CVA", "value"].values[0]

    bars = ax.bar(
        ["Gross CVA", "Net CVA"],
        [gross, net],
        color=["firebrick", "steelblue"],
        width=0.4,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    benefit_pct = netting_df.loc[netting_df["metric"] == "Benefit (%)", "value"].values[0]
    ax.set_title(f"Netting Benefit: {benefit_pct:.1f}% CVA Reduction")
    ax.set_ylabel("CVA Value")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    return fig, ax


def plot_wrong_way_risk(
    correlations: list[float],
    cvas: list[float],
) -> tuple:
    """Plot CVA as a function of exposure-default correlation.

    Parameters
    ----------
    correlations : list of float
        Correlation values.
    cvas : list of float
        CVA values at each correlation level.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(correlations, cvas, marker="o", linewidth=2, color="steelblue", markersize=8)

    # Mark the zero-correlation baseline
    zero_idx = None
    for i, c in enumerate(correlations):
        if abs(c) < 1e-10:
            zero_idx = i
            break

    if zero_idx is not None:
        ax.axhline(
            y=cvas[zero_idx],
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Baseline CVA (rho=0): {cvas[zero_idx]:,.0f}",
        )

    ax.set_xlabel("Exposure-Default Correlation (rho)")
    ax.set_ylabel("CVA")
    ax.set_title("Wrong-Way Risk: CVA vs Exposure-Default Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax
