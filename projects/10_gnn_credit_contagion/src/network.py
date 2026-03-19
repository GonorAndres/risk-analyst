"""Financial network generation and representation.

Generates synthetic interbank networks with realistic features:
- Random exposure topology (Erdos-Renyi style)
- Node features: capital ratio, leverage, asset quality, size, interbank exposure ratio
- Bilateral liability matrix
- External assets per institution

Reference:
- Battiston et al. (2012), "DebtRank: too central to fail?"
- Eisenberg & Noe (2001), "Systemic risk in financial systems"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def generate_financial_network(
    n_nodes: int, n_edges: int, seed: int
) -> dict:
    """Generate a random financial network.

    Creates a synthetic interbank network with weighted exposures,
    node-level financial features, and a bilateral liability matrix.

    Parameters
    ----------
    n_nodes : int
        Number of financial institutions (nodes).
    n_edges : int
        Number of interbank exposures (directed edges).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
        - adjacency : ndarray of shape (n_nodes, n_nodes)
            Weighted adjacency matrix. A[i,j] > 0 means i has exposure to j.
        - node_features : ndarray of shape (n_nodes, 5)
            Features: [capital_ratio, leverage, asset_quality, size,
                        interbank_exposure_ratio]
        - liabilities : ndarray of shape (n_nodes, n_nodes)
            L[i,j] = amount institution i owes to institution j.
        - assets : ndarray of shape (n_nodes,)
            Total external assets per institution.
        - labels : ndarray of shape (n_nodes,)
            Binary default labels (1 = default) based on capital ratio.
    """
    rng = np.random.default_rng(seed)

    # --- Generate directed edges (no self-loops) ---
    max_edges = n_nodes * (n_nodes - 1)
    n_edges = min(n_edges, max_edges)

    # All possible directed pairs excluding self-loops
    all_pairs = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                all_pairs.append((i, j))
    all_pairs = np.array(all_pairs)
    chosen_idx = rng.choice(len(all_pairs), size=n_edges, replace=False)
    edges = all_pairs[chosen_idx]

    # --- Build liability matrix ---
    # Exposure amounts drawn from log-normal to get realistic heavy-tailed sizes
    liabilities = np.zeros((n_nodes, n_nodes))
    exposure_amounts = rng.lognormal(mean=3.0, sigma=1.0, size=n_edges)
    for k, (i, j) in enumerate(edges):
        liabilities[i, j] = exposure_amounts[k]

    # --- Adjacency = liabilities (weighted by exposure) ---
    adjacency = liabilities.copy()

    # --- External assets per node ---
    # Assets should generally exceed total liabilities for most nodes
    total_liabilities_per_node = liabilities.sum(axis=1)
    # Base assets: proportional to liabilities with some buffer
    asset_multiplier = rng.uniform(0.8, 2.5, size=n_nodes)
    base_assets = rng.lognormal(mean=4.0, sigma=0.8, size=n_nodes)
    assets = base_assets + total_liabilities_per_node * asset_multiplier

    # --- Node features ---
    # capital_ratio: (assets - liabilities) / assets, can be negative for distressed
    total_liab = liabilities.sum(axis=1)
    capital_ratio = (assets - total_liab) / np.maximum(assets, 1e-8)
    capital_ratio = np.clip(capital_ratio, -0.5, 1.0)

    # leverage: total_liab / equity, capped
    equity = np.maximum(assets - total_liab, 1e-8)
    leverage = total_liab / equity
    leverage = np.clip(leverage, 0.0, 50.0)

    # asset_quality: fraction of performing interbank claims (random)
    asset_quality = rng.beta(a=5.0, b=2.0, size=n_nodes)

    # size: log of total assets (normalized)
    size_raw = np.log1p(assets)
    size = (size_raw - size_raw.mean()) / (size_raw.std() + 1e-8)

    # interbank_exposure_ratio: interbank claims / total assets
    interbank_claims = liabilities.sum(axis=0)  # what others owe this node
    interbank_exposure_ratio = interbank_claims / np.maximum(assets, 1e-8)
    interbank_exposure_ratio = np.clip(interbank_exposure_ratio, 0.0, 5.0)

    node_features = np.column_stack([
        capital_ratio,
        leverage,
        asset_quality,
        size,
        interbank_exposure_ratio,
    ])

    # --- Default labels based on capital ratio threshold ---
    # Institutions with capital_ratio < 0.1 are considered in default
    labels = (capital_ratio < 0.1).astype(np.float64)

    return {
        "adjacency": adjacency,
        "node_features": node_features,
        "liabilities": liabilities,
        "assets": assets,
        "labels": labels,
    }


def compute_centrality(adjacency: NDArray[np.float64]) -> pd.DataFrame:
    """Compute centrality metrics for each node.

    Computes degree centrality, approximate betweenness centrality,
    and eigenvector centrality from the adjacency matrix.

    Parameters
    ----------
    adjacency : ndarray of shape (n_nodes, n_nodes)
        Weighted adjacency matrix.

    Returns
    -------
    pd.DataFrame
        Columns: node, degree_centrality, betweenness_centrality,
                 eigenvector_centrality.
    """
    n = adjacency.shape[0]

    # --- Degree centrality (normalized) ---
    # Use binary connectivity for degree
    binary_adj = (adjacency > 0).astype(float)
    in_degree = binary_adj.sum(axis=0)
    out_degree = binary_adj.sum(axis=1)
    degree = (in_degree + out_degree) / (2.0 * (n - 1)) if n > 1 else in_degree

    # --- Eigenvector centrality via power iteration ---
    # Use the symmetrized adjacency for eigenvector centrality
    sym_adj = binary_adj + binary_adj.T
    eigvec = np.ones(n) / np.sqrt(n)
    for _ in range(100):
        eigvec_new = sym_adj @ eigvec
        norm = np.linalg.norm(eigvec_new)
        if norm < 1e-12:
            break
        eigvec_new /= norm
        if np.linalg.norm(eigvec_new - eigvec) < 1e-10:
            break
        eigvec = eigvec_new
    eigvec = np.abs(eigvec)
    max_eig = eigvec.max()
    if max_eig > 0:
        eigvec /= max_eig

    # --- Approximate betweenness centrality ---
    # BFS-based shortest path betweenness on unweighted graph
    betweenness = _approx_betweenness(binary_adj)

    return pd.DataFrame({
        "node": np.arange(n),
        "degree_centrality": degree,
        "betweenness_centrality": betweenness,
        "eigenvector_centrality": eigvec,
    })


def _approx_betweenness(binary_adj: NDArray[np.float64]) -> NDArray[np.float64]:
    """Approximate betweenness centrality using BFS from all nodes.

    Uses Brandes' algorithm on the unweighted graph.

    Parameters
    ----------
    binary_adj : ndarray of shape (n, n)
        Binary adjacency matrix.

    Returns
    -------
    ndarray of shape (n,)
        Normalized betweenness centrality.
    """
    n = binary_adj.shape[0]
    betweenness = np.zeros(n)

    for s in range(n):
        # BFS from source s
        stack = []
        predecessors: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
        dist[s] = 0
        queue = [s]
        head = 0

        while head < len(queue):
            v = queue[head]
            head += 1
            stack.append(v)
            neighbors = np.where(binary_adj[v] > 0)[0]
            for w in neighbors:
                # First visit
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                # Shortest path via v
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        # Back-propagation of dependencies
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalize
    norm_factor = (n - 1) * (n - 2) if n > 2 else 1.0
    betweenness /= norm_factor

    return betweenness


def network_stats(adjacency: NDArray[np.float64]) -> dict:
    """Compute basic network statistics.

    Parameters
    ----------
    adjacency : ndarray of shape (n_nodes, n_nodes)
        Weighted adjacency matrix.

    Returns
    -------
    dict
        Keys: n_nodes, n_edges, density, avg_degree, max_degree.
    """
    n = adjacency.shape[0]
    binary_adj = (adjacency > 0).astype(float)
    n_edges = int(binary_adj.sum())
    max_possible = n * (n - 1)
    density = n_edges / max_possible if max_possible > 0 else 0.0

    # Out-degree for directed graph
    out_degree = binary_adj.sum(axis=1)

    return {
        "n_nodes": n,
        "n_edges": n_edges,
        "density": density,
        "avg_degree": float(out_degree.mean()),
        "max_degree": int(out_degree.max()),
    }
