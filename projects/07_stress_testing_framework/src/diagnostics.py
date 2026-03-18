"""Diagnostic visualisations for the stress testing framework.

All plotting functions return ``(fig, ax)`` tuples and never call
``plt.show()`` so that they compose cleanly in notebooks and tests.

References:
    - Fed DFAST 2023, Results Visualization Annex.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scenario_paths(
    scenarios: dict[str, pd.DataFrame],
    factor: str,
) -> tuple:
    """Line plot of a single macro factor across multiple scenarios.

    Parameters
    ----------
    scenarios : dict[str, pd.DataFrame]
        Mapping of scenario name -> DataFrame (indexed by quarter).
    factor : str
        Column name to plot.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in scenarios.items():
        ax.plot(df.index, df[factor], marker="o", label=name)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(factor)
    ax.set_title(f"Scenario Paths: {factor}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_loss_waterfall(sensitivity_df: pd.DataFrame) -> tuple:
    """Waterfall chart showing each factor's contribution to total stressed loss.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Must contain columns ``factor`` and ``elasticity``.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    factors = sensitivity_df["factor"].values
    values = sensitivity_df["elasticity"].values.astype(float)

    cumulative = np.zeros(len(values) + 1)
    for i, v in enumerate(values):
        cumulative[i + 1] = cumulative[i] + v

    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in values]

    for i in range(len(values)):
        bottom = min(cumulative[i], cumulative[i + 1])
        height = abs(values[i])
        ax.bar(factors[i], height, bottom=bottom, color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Loss Contribution")
    ax.set_title("Loss Waterfall: Factor Contributions")
    ax.axhline(y=0, color="black", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig, ax


def plot_historical_comparison(historical_results: pd.DataFrame) -> tuple:
    """Bar chart comparing predicted portfolio loss under each historical crisis.

    Parameters
    ----------
    historical_results : pd.DataFrame
        Must contain columns ``scenario`` and ``predicted_loss``.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = historical_results["scenario"].values
    losses = historical_results["predicted_loss"].values.astype(float)

    bars = ax.bar(scenarios, losses, color="#337ab7", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Predicted Loss")
    ax.set_title("Historical Scenario Comparison")
    ax.axhline(y=0, color="black", linewidth=0.8)

    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{loss:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig, ax


def plot_capital_impact(dfast_results: pd.DataFrame) -> tuple:
    """Bar chart of capital ratio impact by DFAST scenario.

    Parameters
    ----------
    dfast_results : pd.DataFrame
        Must contain columns ``scenario`` and ``capital_ratio_impact``.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = dfast_results["scenario"].values
    ratios = dfast_results["capital_ratio_impact"].values.astype(float)

    colors = ["#5cb85c" if r >= 0.045 else "#f0ad4e" if r >= 0 else "#d9534f" for r in ratios]
    bars = ax.bar(scenarios, ratios * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Post-Stress CET1 Ratio (%)")
    ax.set_title("Capital Impact by Scenario")
    ax.axhline(y=4.5, color="red", linestyle="--", linewidth=1, label="Minimum CET1 (4.5%)")
    ax.legend()

    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig, ax
