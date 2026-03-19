"""Diagnostic visualizations for GNN credit contagion.

All plotting functions return (fig, ax) tuples for composability.
Network layout uses a simple force-directed (Fruchterman-Reingold)
algorithm implemented in numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# matplotlib imports with Agg backend for headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def _fruchterman_reingold(
    adjacency: NDArray[np.float64],
    n_iter: int = 50,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simple force-directed layout (Fruchterman-Reingold).

    Parameters
    ----------
    adjacency : ndarray of shape (n, n)
        Adjacency matrix.
    n_iter : int
        Number of layout iterations.
    seed : int
        Random seed for initial positions.

    Returns
    -------
    ndarray of shape (n, 2)
        2D positions for each node.
    """
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    pos = rng.uniform(-1, 1, size=(n, 2))

    # Optimal distance
    area = 4.0
    k = np.sqrt(area / max(n, 1))
    temp = 1.0

    binary_adj = (adjacency > 0).astype(float)

    for _ in range(n_iter):
        # Repulsive forces between all pairs
        disp = np.zeros((n, 2))
        for i in range(n):
            delta = pos[i] - pos  # (n, 2)
            dist = np.linalg.norm(delta, axis=1)
            dist = np.maximum(dist, 0.01)
            # Repulsive force: k^2 / dist
            force = k * k / dist
            force[i] = 0  # no self-force
            disp[i] = np.sum(
                (delta.T * force).T, axis=0
            )

        # Attractive forces along edges
        edges = np.argwhere(binary_adj > 0)
        for i, j in edges:
            delta = pos[i] - pos[j]
            dist = max(np.linalg.norm(delta), 0.01)
            # Attractive force: dist / k
            force_mag = dist / k
            force_dir = delta / dist
            disp[i] -= force_mag * force_dir
            disp[j] += force_mag * force_dir

        # Apply displacement with temperature
        disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp_norm = np.maximum(disp_norm, 0.01)
        pos += (disp / disp_norm) * np.minimum(disp_norm, temp)

        # Cool down
        temp *= 0.95

    # Normalize to [0, 1]
    pos -= pos.min(axis=0)
    extent = pos.max(axis=0) - pos.min(axis=0)
    extent = np.maximum(extent, 1e-8)
    pos /= extent

    return pos


def plot_network(
    adjacency: NDArray[np.float64],
    node_colors: NDArray[np.float64],
    title: str = "Financial Network",
) -> tuple[Figure, Axes]:
    """Visualize the financial network with spring layout.

    Nodes are colored by the provided array (e.g., risk level or default prob).

    Parameters
    ----------
    adjacency : ndarray of shape (n, n)
        Weighted adjacency matrix.
    node_colors : ndarray of shape (n,)
        Values used to color nodes (e.g., default probability).
    title : str
        Plot title.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    n = adjacency.shape[0]
    pos = _fruchterman_reingold(adjacency)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Draw edges
    edges = np.argwhere(adjacency > 0)
    for i, j in edges:
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color="gray",
            alpha=0.2,
            linewidth=0.5,
        )

    # Draw nodes
    scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=node_colors,
        cmap="RdYlGn_r",
        s=80,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    plt.colorbar(scatter, ax=ax, label="Risk level")

    # Node labels
    for i in range(n):
        ax.annotate(
            str(i),
            (pos[i, 0], pos[i, 1]),
            fontsize=6,
            ha="center",
            va="center",
            zorder=10,
        )

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()

    return fig, ax


def plot_cascade_size(cascade_df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Bar chart showing number of defaults per shocked node.

    Parameters
    ----------
    cascade_df : pd.DataFrame
        Must have columns: shocked_node, n_defaults.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Sort by cascade size for readability
    df_sorted = cascade_df.sort_values("n_defaults", ascending=False)

    ax.bar(
        range(len(df_sorted)),
        df_sorted["n_defaults"],
        color="steelblue",
        edgecolor="navy",
        alpha=0.8,
    )
    ax.set_xlabel("Shocked Node (sorted by cascade size)")
    ax.set_ylabel("Number of Defaults")
    ax.set_title("Cascade Size by Shocked Node")

    # Only label x-axis every few ticks for readability
    n_ticks = min(20, len(df_sorted))
    tick_indices = np.linspace(0, len(df_sorted) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        df_sorted["shocked_node"].values[tick_indices],
        rotation=45,
        fontsize=8,
    )

    fig.tight_layout()
    return fig, ax


def plot_debtrank_distribution(
    debtrank: NDArray[np.float64],
) -> tuple[Figure, Axes]:
    """Histogram of DebtRank values across all nodes.

    Parameters
    ----------
    debtrank : ndarray of shape (n,)
        DebtRank for each node.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(
        debtrank,
        bins=20,
        color="coral",
        edgecolor="darkred",
        alpha=0.8,
    )
    ax.axvline(
        np.mean(debtrank),
        color="black",
        linestyle="--",
        label=f"Mean = {np.mean(debtrank):.3f}",
    )
    ax.set_xlabel("DebtRank")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of DebtRank (Systemic Importance)")
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_gcn_predictions(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> tuple[Figure, Axes]:
    """Scatter plot of true labels vs GCN predicted default probability.

    Parameters
    ----------
    y_true : ndarray of shape (n,)
        True binary labels.
    y_pred : ndarray of shape (n,)
        Predicted default probabilities.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color by true label
    colors = ["green" if label == 0 else "red" for label in y_true]
    ax.scatter(
        np.arange(len(y_true)),
        y_pred,
        c=colors,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold = 0.5")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Predicted Default Probability")
    ax.set_title("GCN Predictions vs True Labels")
    ax.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="green",
                label="Non-default", markersize=8,
            ),
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="red",
                label="Default", markersize=8,
            ),
        ]
    )

    fig.tight_layout()
    return fig, ax
