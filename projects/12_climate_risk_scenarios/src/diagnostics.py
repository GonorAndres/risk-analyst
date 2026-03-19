"""Diagnostic visualizations for climate risk analysis.

Provides heatmaps, tornado charts, line plots, and bar charts for
scenario comparison, Sobol sensitivity, WACI evolution, and stranded
asset exposure.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scenario_heatmap(loss_df: pd.DataFrame) -> tuple:
    """Plot a sector x scenario heatmap of losses.

    Parameters
    ----------
    loss_df : pd.DataFrame
        DataFrame with columns: scenario, year, transition_loss,
        physical_loss, total_loss. Typically from scenario_comparison().

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    # Pivot to scenario x year for total_loss
    pivot = loss_df.pivot(index="scenario", columns="year", values="total_loss")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=8, color="black" if val < pivot.values.max() * 0.7 else "white")

    ax.set_title("Climate Risk: Total Loss by Scenario and Horizon")
    ax.set_xlabel("Horizon Year")
    ax.set_ylabel("Scenario")
    fig.colorbar(im, ax=ax, label="Portfolio Loss (fraction)")
    fig.tight_layout()

    return fig, ax


def plot_sobol_tornado(sobol_df: pd.DataFrame) -> tuple:
    """Plot horizontal bar chart of Sobol indices.

    S1 (first-order) in blue, ST (total-order) in red.

    Parameters
    ----------
    sobol_df : pd.DataFrame
        DataFrame with columns: factor, S1, ST.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(sobol_df))
    bar_height = 0.35

    ax.barh(y_pos - bar_height / 2, sobol_df["S1"].values,
            height=bar_height, color="#4472C4", label="S1 (First-order)")
    ax.barh(y_pos + bar_height / 2, sobol_df["ST"].values,
            height=bar_height, color="#C44E52", label="ST (Total-order)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sobol_df["factor"].values)
    ax.set_xlabel("Sobol Index")
    ax.set_title("Sobol Sensitivity Analysis: Climate Risk Factors")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(sobol_df["ST"].max() * 1.15, 0.1))
    fig.tight_layout()

    return fig, ax


def plot_waci_evolution(waci_df: pd.DataFrame) -> tuple:
    """Plot WACI evolution over time, one line per scenario.

    Parameters
    ----------
    waci_df : pd.DataFrame
        DataFrame with columns: year, waci, scenario.
        If 'scenario' is missing, plots a single line.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if "scenario" in waci_df.columns:
        for scenario, group in waci_df.groupby("scenario"):
            ax.plot(group["year"], group["waci"], label=scenario, linewidth=1.5)
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.plot(waci_df["year"], waci_df["waci"], linewidth=2, color="#4472C4")

    ax.set_xlabel("Year")
    ax.set_ylabel("WACI (tCO2/$M revenue)")
    ax.set_title("Weighted Average Carbon Intensity Evolution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_stranded_assets(
    sector_data: pd.DataFrame,
    carbon_prices: list[float],
) -> tuple:
    """Plot stranded asset fraction by sector at different carbon prices.

    Parameters
    ----------
    sector_data : pd.DataFrame
        Indexed by sector with extraction_cost and reserves_value columns.
    carbon_prices : list[float]
        Carbon prices to evaluate.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes.
    """
    from transition_risk import stranded_asset_exposure

    # Filter to sectors with reserves
    has_reserves = sector_data[sector_data["reserves_value"] > 0]
    if has_reserves.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No sectors with reserves", transform=ax.transAxes,
                ha="center", va="center")
        return fig, ax

    sectors = has_reserves.index.tolist()
    n_sectors = len(sectors)
    n_prices = len(carbon_prices)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / n_prices
    x = np.arange(n_sectors)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_prices))

    for j, cp in enumerate(carbon_prices):
        fractions = stranded_asset_exposure(
            has_reserves["reserves_value"].values,
            has_reserves["extraction_cost"].values,
            carbon_price=cp,
        )
        offset = (j - n_prices / 2 + 0.5) * bar_width
        ax.bar(x + offset, fractions, width=bar_width,
               color=colors[j], label=f"${cp:.0f}/tCO2")

    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=45, ha="right")
    ax.set_ylabel("Stranded Fraction")
    ax.set_title("Stranded Asset Exposure by Sector and Carbon Price")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    return fig, ax
