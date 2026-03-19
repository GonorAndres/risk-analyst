"""Diagnostic plots for conformal risk prediction.

Provides coverage comparison, interval width analysis, ACI trajectory,
and PD interval visualization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_coverage_over_time(
    coverage_conformal: np.ndarray,
    coverage_parametric: np.ndarray,
    title: str = "Rolling Coverage: Conformal vs Parametric",
) -> plt.Figure:
    """Plot rolling coverage comparison between conformal and parametric.

    Parameters
    ----------
    coverage_conformal : np.ndarray
        Rolling coverage from conformal method.
    coverage_parametric : np.ndarray
        Rolling coverage from parametric method.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(len(coverage_conformal))
    ax.plot(t, coverage_conformal, label="Conformal", linewidth=1.5)
    ax.plot(t, coverage_parametric, label="Parametric", linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.6, label="90% target")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative coverage")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_interval_width_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing interval widths across methods.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: method, coverage, avg_width, median_width.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = comparison_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    # Coverage
    axes[0].bar(x, comparison_df["coverage"], width, color="steelblue")
    axes[0].axhline(y=0.9, color="red", linestyle="--", alpha=0.6, label="90% target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Empirical Coverage by Method")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Width
    axes[1].bar(
        x - width / 2,
        comparison_df["avg_width"],
        width,
        label="Mean width",
        color="steelblue",
    )
    axes[1].bar(
        x + width / 2,
        comparison_df["median_width"],
        width,
        label="Median width",
        color="coral",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Interval width")
    axes[1].set_title("Interval Width by Method")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_adaptive_coverage(aci_results: dict, alpha: float = 0.1) -> plt.Figure:
    """Plot ACI coverage trajectory with regime change annotation.

    Parameters
    ----------
    aci_results : dict
        Output from ``run_aci_experiment`` or ``ConformalRiskModel.run_adaptive``.
    alpha : float
        Target miscoverage level.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    coverage = aci_results["coverage_trajectory"]
    thresholds = aci_results["thresholds"]
    t = np.arange(len(coverage))

    n_calm = aci_results.get("n_calm", len(coverage) // 2)

    # Top: coverage trajectory
    axes[0].plot(t, coverage, label="Running coverage", linewidth=1.2)
    axes[0].axhline(
        y=1 - alpha, color="red", linestyle="--", alpha=0.6, label=f"{1 - alpha:.0%} target"
    )
    axes[0].axvline(
        x=n_calm, color="orange", linestyle=":", alpha=0.8, label="Regime change"
    )
    axes[0].set_ylabel("Cumulative coverage")
    axes[0].set_title("Adaptive Conformal Inference -- Coverage Trajectory")
    axes[0].legend(loc="lower left")
    axes[0].set_ylim(0.5, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Bottom: adaptive thresholds
    axes[1].plot(t, thresholds[: len(t)], linewidth=1.2, color="green")
    axes[1].axvline(
        x=n_calm, color="orange", linestyle=":", alpha=0.8, label="Regime change"
    )
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Adaptive threshold")
    axes[1].set_title("Adaptive Threshold Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_conformal_pd(
    pd_lower: np.ndarray,
    pd_upper: np.ndarray,
    pd_point: np.ndarray,
    y_true: np.ndarray,
    n_show: int = 100,
) -> plt.Figure:
    """Plot conformal PD predictions with confidence bands.

    Parameters
    ----------
    pd_lower : np.ndarray
        Lower bound of PD interval.
    pd_upper : np.ndarray
        Upper bound of PD interval.
    pd_point : np.ndarray
        Point estimate of PD.
    y_true : np.ndarray
        True default indicators (0/1).
    n_show : int
        Number of observations to display.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    n_show = min(n_show, len(pd_point))
    idx = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Conformal band
    ax.fill_between(
        idx,
        pd_lower[:n_show],
        pd_upper[:n_show],
        alpha=0.3,
        color="steelblue",
        label="Conformal interval",
    )
    ax.plot(idx, pd_point[:n_show], "b-", linewidth=1, label="PD point estimate")

    # Mark true defaults
    defaults = np.where(y_true[:n_show] == 1)[0]
    non_defaults = np.where(y_true[:n_show] == 0)[0]
    ax.scatter(
        defaults,
        y_true[:n_show][defaults],
        color="red",
        marker="x",
        s=40,
        zorder=5,
        label="Default",
    )
    ax.scatter(
        non_defaults,
        y_true[:n_show][non_defaults],
        color="green",
        marker=".",
        s=20,
        zorder=5,
        alpha=0.4,
        label="Non-default",
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel("Probability of Default")
    ax.set_title("Conformal PD Intervals with True Default Status")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
