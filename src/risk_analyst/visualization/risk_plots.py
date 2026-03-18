"""Risk visualization utilities.

Provides publication-quality plots for VaR backtesting, rolling volatility,
correlation heatmaps, and loss distribution analysis.

References:
    - Jorion (2007), Ch. 6: backtesting visualizations.
    - Basel Committee (1996): traffic-light zone shading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def plot_var_backtest(
    losses: pd.Series | np.ndarray,
    var_series: pd.Series | np.ndarray,
    title: str = "VaR Backtesting",
    figsize: tuple[int, int] = (14, 6),
) -> Figure:
    """Time series of losses with VaR overlay; violations highlighted in red.

    Parameters
    ----------
    losses : pd.Series | np.ndarray
        Realized portfolio losses.
    var_series : pd.Series | np.ndarray
        VaR forecast series (same length as *losses*).
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure dimensions.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    losses_arr = np.asarray(losses)
    var_arr = np.asarray(var_series)
    violations = losses_arr >= var_arr

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(losses_arr))
    if hasattr(losses, "index"):
        x = losses.index  # type: ignore[assignment]

    ax.plot(x, losses_arr, color="steelblue", linewidth=0.7, alpha=0.8, label="Portfolio Loss")
    ax.plot(x, var_arr, color="firebrick", linewidth=1.2, linestyle="--", label="VaR")

    # Highlight violations
    viol_idx = np.where(violations)[0]
    if hasattr(losses, "index"):
        viol_x = losses.index[viol_idx]  # type: ignore[index]
    else:
        viol_x = viol_idx
    ax.scatter(
        viol_x,
        losses_arr[viol_idx],
        color="red",
        s=20,
        zorder=5,
        label=f"Violations ({len(viol_idx)})",
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rolling_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int = 21,
    figsize: tuple[int, int] = (14, 5),
) -> Figure:
    """Rolling standard deviation (annualized) of returns.

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series
        Asset or portfolio returns.
    window : int
        Rolling window in trading days.
    figsize : tuple[int, int]
        Figure dimensions.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    import pandas as pd  # noqa: F811

    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    rolling_vol = returns.rolling(window).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=figsize)
    for col in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[col], linewidth=0.9, label=str(col))

    ax.set_title(f"Rolling {window}-Day Volatility (Annualized)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    figsize: tuple[int, int] = (8, 6),
) -> Figure:
    """Correlation matrix heatmap.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with one column per asset.
    figsize : tuple[int, int]
        Figure dimensions.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    corr = returns.corr()
    n = len(corr)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if abs(corr.values[i, j]) > 0.6 else "black",
            )

    ax.set_title("Return Correlation Matrix", fontsize=13)
    fig.tight_layout()
    return fig


def plot_loss_distribution(
    losses: np.ndarray,
    var: float,
    es: float,
    title: str = "Loss Distribution",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Histogram of losses with VaR and ES vertical lines.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of portfolio losses.
    var : float
        Value-at-Risk estimate.
    es : float
        Expected Shortfall estimate.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure dimensions.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(losses, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(var, color="firebrick", linewidth=2, linestyle="--", label=f"VaR = {var:.4f}")
    ax.axvline(es, color="darkorange", linewidth=2, linestyle="-.", label=f"ES = {es:.4f}")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
