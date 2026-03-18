"""Diagnostic plots for copula dependency analysis.

Produces scatter plots of pseudo-uniform data, simulated-vs-empirical
comparisons, tail dependence bar charts, and portfolio VaR comparisons.

All plot functions return ``(fig, ax)`` tuples following the repository's
matplotlib convention.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_copula_scatter(
    u_data: np.ndarray,
    title: str = "Copula Scatter",
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of bivariate pseudo-uniform data.

    Shows the dependence structure in the unit square [0, 1]^2.

    Parameters
    ----------
    u_data : np.ndarray
        Uniform data of shape (n, 2).
    title : str
        Plot title.

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u_data[:, 0], u_data[:, 1], s=3, alpha=0.3, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$u_2$")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


def plot_simulated_vs_empirical(
    u_empirical: np.ndarray,
    u_simulated: np.ndarray,
    family: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Side-by-side scatter: empirical vs simulated copula data.

    Parameters
    ----------
    u_empirical : np.ndarray
        Empirical pseudo-uniform data, shape (n, 2).
    u_simulated : np.ndarray
        Simulated data from the fitted copula, shape (m, 2).
    family : str
        Copula family name (for labelling).

    Returns
    -------
    tuple[Figure, ndarray[Axes]]
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        u_empirical[:, 0], u_empirical[:, 1],
        s=3, alpha=0.3, color="steelblue",
    )
    axes[0].set_title("Empirical")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("$u_1$")
    axes[0].set_ylabel("$u_2$")
    axes[0].set_aspect("equal")

    axes[1].scatter(
        u_simulated[:, 0], u_simulated[:, 1],
        s=3, alpha=0.3, color="coral",
    )
    axes[1].set_title(f"Simulated ({family.capitalize()})")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("$u_1$")
    axes[1].set_ylabel("$u_2$")
    axes[1].set_aspect("equal")

    fig.suptitle(f"Empirical vs {family.capitalize()} Copula", fontsize=13)
    fig.tight_layout()
    return fig, axes


def plot_tail_dependence_comparison(
    comparison_df: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart of lower and upper tail dependence by family.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Must contain columns ``family``, ``lambda_L``, ``lambda_U``.

    Returns
    -------
    tuple[Figure, Axes]
    """
    families = comparison_df["family"].values
    lambda_L = comparison_df["lambda_L"].values
    lambda_U = comparison_df["lambda_U"].values

    x = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_L = ax.bar(x - width / 2, lambda_L, width, label=r"$\lambda_L$", color="steelblue")
    bars_U = ax.bar(x + width / 2, lambda_U, width, label=r"$\lambda_U$", color="coral")

    ax.set_xlabel("Copula Family")
    ax.set_ylabel("Tail Dependence Coefficient")
    ax.set_title("Tail Dependence by Copula Family")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families])
    ax.legend()
    ax.set_ylim(0, max(max(lambda_L), max(lambda_U)) * 1.3 + 0.02)

    fig.tight_layout()
    return fig, ax


def plot_var_by_copula(
    var_df: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart of portfolio VaR by copula family and alpha.

    Parameters
    ----------
    var_df : pd.DataFrame
        DataFrame with a ``family`` column and ``VaR_<alpha>`` columns.

    Returns
    -------
    tuple[Figure, Axes]
    """
    families = var_df["family"].values
    var_cols = [c for c in var_df.columns if c.startswith("VaR_")]
    n_alphas = len(var_cols)
    n_families = len(families)

    x = np.arange(n_families)
    total_width = 0.7
    bar_width = total_width / n_alphas

    colors = ["steelblue", "coral", "forestgreen", "goldenrod"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(var_cols):
        offset = (i - n_alphas / 2 + 0.5) * bar_width
        alpha_label = col.replace("VaR_", "")
        ax.bar(
            x + offset,
            var_df[col].values,
            bar_width,
            label=f"VaR {alpha_label}",
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Copula Family")
    ax.set_ylabel("Portfolio VaR (loss)")
    ax.set_title("Portfolio VaR by Copula Family")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families])
    ax.legend()

    fig.tight_layout()
    return fig, ax
