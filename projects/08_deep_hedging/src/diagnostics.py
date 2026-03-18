"""Diagnostic plots for deep hedging.

All functions return (fig, ax) tuples for composability and testing.

Reference: Buehler et al. (2019), Figures 2-5.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_loss_history(losses: list[float]) -> tuple:
    """Plot training loss (risk measure) over epochs.

    Parameters
    ----------
    losses : list of float
        Risk measure value at each epoch.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Risk Measure")
    ax.set_title("Deep Hedging Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_pnl_distribution(
    pnl_bs: NDArray[np.float64],
    pnl_deep: NDArray[np.float64],
) -> tuple:
    """Plot overlaid histograms of hedging P&L for BS and deep hedge.

    Parameters
    ----------
    pnl_bs : ndarray
        P&L from Black-Scholes delta hedging.
    pnl_deep : ndarray
        P&L from deep hedging.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(pnl_bs.min(), pnl_deep.min()),
        max(pnl_bs.max(), pnl_deep.max()),
        60,
    )

    ax.hist(pnl_bs, bins=bins, alpha=0.5, label="BS Delta Hedge", color="steelblue", density=True)
    ax.hist(pnl_deep, bins=bins, alpha=0.5, label="Deep Hedge", color="darkorange", density=True)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Hedging P&L")
    ax.set_ylabel("Density")
    ax.set_title("P&L Distribution: BS Delta vs Deep Hedge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_hedge_ratio_comparison(
    S_range: NDArray[np.float64],
    bs_delta: NDArray[np.float64],
    deep_delta: NDArray[np.float64],
    t: float,
) -> tuple:
    """Plot hedge ratio (delta) as a function of stock price.

    Parameters
    ----------
    S_range : ndarray
        Stock price values.
    bs_delta : ndarray
        Black-Scholes delta values.
    deep_delta : ndarray
        Deep hedging delta values.
    t : float
        Time point (for title annotation).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(S_range, bs_delta, label="BS Delta", linewidth=2, color="steelblue")
    ax.plot(
        S_range,
        deep_delta,
        label="Deep Hedge Delta",
        linewidth=2,
        linestyle="--",
        color="darkorange",
    )

    ax.set_xlabel("Stock Price S")
    ax.set_ylabel("Hedge Ratio (Delta)")
    ax.set_title(f"Hedge Ratio Comparison at t = {t:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    return fig, ax


def plot_transaction_cost_impact(
    cost_rates: list[float],
    bs_cvars: list[float],
    deep_cvars: list[float],
) -> tuple:
    """Plot CVaR vs transaction cost rate for BS and deep hedge.

    Parameters
    ----------
    cost_rates : list of float
        Transaction cost rates.
    bs_cvars : list of float
        CVaR (95%) for BS delta hedging at each cost rate.
    deep_cvars : list of float
        CVaR (95%) for deep hedging at each cost rate.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    cost_bps = [c * 10000 for c in cost_rates]  # convert to basis points

    ax.plot(
        cost_bps,
        bs_cvars,
        marker="o",
        linewidth=2,
        label="BS Delta Hedge",
        color="steelblue",
    )
    ax.plot(
        cost_bps,
        deep_cvars,
        marker="s",
        linewidth=2,
        label="Deep Hedge",
        color="darkorange",
    )

    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("CVaR (95%)")
    ax.set_title("Transaction Cost Impact on Hedging Risk")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
