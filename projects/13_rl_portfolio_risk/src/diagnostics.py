"""Diagnostic plots for RL portfolio risk management.

All functions return (fig, ax) tuples for composability and testing.

Reference: Follows the plotting pattern from P08 deep hedging diagnostics.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def plot_cumulative_returns(
    results: dict[str, NDArray[np.float64]],
) -> tuple:
    """Plot cumulative returns for multiple strategies.

    Parameters
    ----------
    results : dict
        Mapping of strategy name -> cumulative returns array.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["steelblue", "darkorange", "forestgreen", "firebrick", "purple"]

    for idx, (name, cum_ret) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        ax.plot(cum_ret, label=name, linewidth=1.5, color=color)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Returns: Strategy Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_allocation_evolution(
    weights: NDArray[np.float64],
    asset_names: list[str],
) -> tuple:
    """Plot portfolio weight evolution as a stacked area chart.

    Parameters
    ----------
    weights : ndarray of shape (T, n_assets)
        Time series of portfolio weights.
    asset_names : list of str
        Names for each asset.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    T = weights.shape[0]
    t = np.arange(T)

    colors = ["steelblue", "darkorange", "forestgreen", "firebrick", "mediumpurple"]
    # Extend colors if needed
    while len(colors) < weights.shape[1]:
        colors.append("gray")

    ax.stackplot(
        t,
        weights.T,
        labels=asset_names,
        colors=colors[: weights.shape[1]],
        alpha=0.8,
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("RL Agent: Allocation Evolution")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_risk_return_scatter(
    comparison_df: pd.DataFrame,
) -> tuple:
    """Scatter plot of CVaR (x) vs total return (y) for each strategy.

    Parameters
    ----------
    comparison_df : DataFrame
        Must contain columns: strategy, total_return, cvar_95.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["steelblue", "darkorange", "forestgreen", "firebrick", "purple"]

    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        color = colors[idx % len(colors)]
        ax.scatter(
            row["cvar_95"],
            row["total_return"],
            s=120,
            color=color,
            zorder=5,
            edgecolors="black",
            linewidth=0.8,
        )
        ax.annotate(
            row["strategy"],
            (row["cvar_95"], row["total_return"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_xlabel("CVaR (95%) -- Tail Risk")
    ax.set_ylabel("Total Return")
    ax.set_title("Risk-Return Trade-off by Strategy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_drawdown_comparison(
    drawdowns: dict[str, NDArray[np.float64]],
) -> tuple:
    """Plot drawdown curves for multiple strategies.

    Parameters
    ----------
    drawdowns : dict
        Mapping of strategy name -> drawdown array (non-negative values).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["steelblue", "darkorange", "forestgreen", "firebrick", "purple"]

    for idx, (name, dd) in enumerate(drawdowns.items()):
        color = colors[idx % len(colors)]
        ax.fill_between(
            np.arange(len(dd)),
            -dd,
            alpha=0.3,
            color=color,
        )
        ax.plot(-dd, label=name, linewidth=1.2, color=color)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
