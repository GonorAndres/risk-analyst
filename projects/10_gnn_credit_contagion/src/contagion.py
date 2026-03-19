"""Cascade and contagion simulation for financial networks.

Implements the Eisenberg-Noe clearing mechanism for finding equilibrium
payment vectors in interbank networks, and the DebtRank algorithm for
measuring systemic importance of individual institutions.

Reference:
- Eisenberg & Noe (2001), "Systemic risk in financial systems",
  Management Science 47(2), 236-249.
- Battiston et al. (2012), "DebtRank: too central to fail?",
  Scientific Reports 2, 541.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def eisenberg_noe_clearing(
    liabilities: NDArray[np.float64],
    assets: NDArray[np.float64],
    max_iter: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find the clearing payment vector via Eisenberg-Noe fixed-point.

    Each institution i has external assets a_i and owes L[i,j] to j.
    Total obligations: p_bar_i = sum_j L[i,j].
    In equilibrium, institution i pays:
        p_i = min(p_bar_i, a_i + sum_j (L[j,i] * p_j / p_bar_j))
    i.e., the minimum of what it owes and what it can afford.

    The algorithm iterates this fixed-point map until convergence.

    Parameters
    ----------
    liabilities : ndarray of shape (n, n)
        L[i,j] = amount institution i owes to institution j.
    assets : ndarray of shape (n,)
        External assets for each institution.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    payments : ndarray of shape (n,)
        Clearing payment vector. payments[i] = total amount i actually pays.
    defaults : ndarray of shape (n,)
        Binary default indicator. defaults[i] = 1 if i cannot pay full obligations.
    """
    n = liabilities.shape[0]
    assets = assets.copy()

    # Total obligations per node
    p_bar = liabilities.sum(axis=1)  # (n,)

    # Relative liability matrix: Pi[i,j] = L[i,j] / p_bar[i]
    # Fraction of i's payments that goes to j
    pi_matrix = np.zeros_like(liabilities)
    for i in range(n):
        if p_bar[i] > 0:
            pi_matrix[i, :] = liabilities[i, :] / p_bar[i]

    # Initialize payments at full obligations (optimistic start)
    payments = p_bar.copy()

    for _ in range(max_iter):
        # Income from other institutions' payments to this node:
        # income_i = sum_j (payments_j * Pi[j,i])
        # Pi[j,i] = L[j,i] / p_bar[j] = fraction j pays to i
        income = pi_matrix.T @ payments  # (n,)

        # Total available resources
        total_resources = assets + income

        # New payments: min of obligations and resources
        payments_new = np.minimum(p_bar, total_resources)
        payments_new = np.maximum(payments_new, 0.0)

        # Check convergence
        if np.max(np.abs(payments_new - payments)) < 1e-10:
            payments = payments_new
            break
        payments = payments_new

    # Default: institution cannot meet full obligations
    defaults = (payments < p_bar - 1e-8).astype(np.float64)

    return payments, defaults


def simulate_cascade(
    liabilities: NDArray[np.float64],
    assets: NDArray[np.float64],
    shocked_nodes: list[int],
    shock_fraction: float,
) -> dict:
    """Simulate a credit contagion cascade.

    Shocks specific nodes by reducing their external assets, then runs
    the Eisenberg-Noe clearing to find the new equilibrium. The cascade
    propagates through the liability network as defaults reduce payments
    to creditors.

    Parameters
    ----------
    liabilities : ndarray of shape (n, n)
        Bilateral liability matrix.
    assets : ndarray of shape (n,)
        External assets per institution.
    shocked_nodes : list of int
        Indices of institutions to shock.
    shock_fraction : float
        Fraction of assets lost by shocked nodes (0 to 1).

    Returns
    -------
    dict
        Keys:
        - n_defaults : int, total number of defaulting institutions.
        - default_nodes : list of int, indices of defaulting institutions.
        - total_loss : float, total asset loss in the system.
        - rounds : int, number of Eisenberg-Noe iterations until convergence.
    """
    assets_shocked = assets.copy()

    # Apply shock
    for node in shocked_nodes:
        assets_shocked[node] *= (1.0 - shock_fraction)

    # Run clearing with higher iteration count to track rounds
    n = liabilities.shape[0]
    p_bar = liabilities.sum(axis=1)

    pi_matrix = np.zeros_like(liabilities)
    for i in range(n):
        if p_bar[i] > 0:
            pi_matrix[i, :] = liabilities[i, :] / p_bar[i]

    payments = p_bar.copy()
    rounds = 0

    for iteration in range(200):
        income = pi_matrix.T @ payments
        total_resources = assets_shocked + income
        payments_new = np.minimum(p_bar, total_resources)
        payments_new = np.maximum(payments_new, 0.0)

        rounds = iteration + 1

        if np.max(np.abs(payments_new - payments)) < 1e-10:
            payments = payments_new
            break
        payments = payments_new

    defaults = (payments < p_bar - 1e-8).astype(np.float64)
    default_nodes = list(np.where(defaults > 0.5)[0])

    # Total loss: difference between promised and actual payments
    total_loss = float(np.sum(p_bar - payments))

    return {
        "n_defaults": int(defaults.sum()),
        "default_nodes": default_nodes,
        "total_loss": total_loss,
        "rounds": rounds,
    }


def compute_debtrank(
    liabilities: NDArray[np.float64],
    assets: NDArray[np.float64],
    shocked_node: int,
) -> float:
    """Compute DebtRank of a single node.

    DebtRank measures the fraction of total economic value in the system
    that is lost when a single institution becomes distressed. It uses
    a linear contagion model where distress propagates proportionally
    to exposure.

    Parameters
    ----------
    liabilities : ndarray of shape (n, n)
        Bilateral liability matrix.
    assets : ndarray of shape (n,)
        External assets per institution.
    shocked_node : int
        Index of the institution to shock.

    Returns
    -------
    float
        DebtRank value in [0, 1]. Higher = more systemically important.
    """
    n = liabilities.shape[0]

    # Economic value of each node: its equity = assets - liabilities
    equity = assets - liabilities.sum(axis=1)
    equity = np.maximum(equity, 1e-8)
    total_value = equity.sum()

    # Impact matrix: W[i,j] = L[j,i] / equity[i]
    # Impact of j's distress on i: how much of i's equity is exposed to j
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if equity[i] > 0 and liabilities[j, i] > 0:
                W[i, j] = min(liabilities[j, i] / equity[i], 1.0)

    # DebtRank iteration
    # h[i] = level of distress of node i, in [0, 1]
    h = np.zeros(n)
    h[shocked_node] = 1.0  # fully distressed

    # State: 0=inactive, 1=distressed, 2=inactive (already propagated)
    state = np.zeros(n, dtype=int)
    state[shocked_node] = 1

    max_rounds = n + 1
    for _ in range(max_rounds):
        h_new = h.copy()
        state_new = state.copy()

        for i in range(n):
            if state[i] == 0:  # undistressed nodes can become distressed
                stress = 0.0
                for j in range(n):
                    if state[j] == 1:  # j is currently distressed
                        stress += W[i, j] * h[j]
                h_new[i] = min(stress, 1.0)
                if h_new[i] > 0:
                    state_new[i] = 1

        # Nodes that were distressed in previous round become inactive
        for i in range(n):
            if state[i] == 1 and state_new[i] == 1:
                # Was already distressed -- mark as propagated
                if i != shocked_node or _ > 0:
                    state_new[i] = 2

        # Check convergence
        if np.array_equal(h_new, h) and np.array_equal(state_new, state):
            break

        # Only newly distressed nodes keep state=1
        for i in range(n):
            if state[i] == 1:
                state_new[i] = 2

        h = h_new
        state = state_new

    # DebtRank = fraction of total value lost (excluding initially shocked node)
    value_lost = 0.0
    for i in range(n):
        if i != shocked_node:
            value_lost += h[i] * equity[i]

    debtrank = value_lost / total_value if total_value > 0 else 0.0
    return float(np.clip(debtrank, 0.0, 1.0))


def systemic_importance(
    liabilities: NDArray[np.float64],
    assets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute DebtRank for every node in the network.

    Parameters
    ----------
    liabilities : ndarray of shape (n, n)
        Bilateral liability matrix.
    assets : ndarray of shape (n,)
        External assets per institution.

    Returns
    -------
    ndarray of shape (n,)
        DebtRank of each node.
    """
    n = liabilities.shape[0]
    dr = np.zeros(n)
    for i in range(n):
        dr[i] = compute_debtrank(liabilities, assets, i)
    return dr
