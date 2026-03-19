"""Training loop for the GCN using evolutionary strategies.

Since the GCN is implemented in pure numpy (no autograd), we use
Natural Evolution Strategies (NES) with mirror sampling for
derivative-free optimisation. Same approach as Project 08 (Deep Hedging).

The NES gradient estimator:
    grad ~= (1 / (2 * pop_size * sigma)) * sum_i (f(theta + sigma*eps_i)
             - f(theta - sigma*eps_i)) * eps_i

Reference:
- Salimans et al. (2017), "Evolution Strategies as a Scalable Alternative
  to Reinforcement Learning", arXiv:1703.03864.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gcn import GCN


def binary_cross_entropy(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Binary cross-entropy loss with numerical stability.

    BCE = -mean( y * log(p) + (1-y) * log(1-p) )

    Parameters
    ----------
    y_true : ndarray of shape (n,) or (n, 1)
        Ground truth binary labels.
    y_pred : ndarray of shape (n,) or (n, 1)
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Non-negative BCE loss.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Clip for numerical stability
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    bce = -np.mean(
        y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
    )
    return float(bce)


def train_gcn(
    gcn: GCN,
    X: NDArray[np.float64],
    A: NDArray[np.float64],
    y: NDArray[np.float64],
    n_epochs: int,
    lr: float,
    seed: int,
    population_size: int = 40,
    noise_std: float = 0.1,
) -> list[float]:
    """Train GCN using Natural Evolution Strategies (NES).

    At each epoch:
    1. Sample perturbation vectors from N(0, I).
    2. Evaluate BCE loss for theta + sigma*eps and theta - sigma*eps
       (mirror sampling for variance reduction).
    3. Estimate gradient via NES formula.
    4. Update parameters with gradient descent.

    Parameters
    ----------
    gcn : GCN
        The graph convolutional network to train.
    X : ndarray of shape (n_nodes, n_features)
        Node feature matrix.
    A : ndarray of shape (n_nodes, n_nodes)
        Adjacency matrix.
    y : ndarray of shape (n_nodes,)
        Binary target labels.
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    seed : int
        Random seed.
    population_size : int
        Number of perturbation samples per epoch.
    noise_std : float
        Standard deviation of perturbation noise.

    Returns
    -------
    list of float
        Loss at each epoch.
    """
    rng = np.random.default_rng(seed)
    theta = gcn.get_flat_params()
    n_params = len(theta)
    sigma = noise_std

    loss_history: list[float] = []

    for epoch in range(n_epochs):
        # Sample perturbations
        epsilon = rng.standard_normal((population_size, n_params))

        # Evaluate loss for each perturbation (mirror sampling)
        losses_plus = np.zeros(population_size)
        losses_minus = np.zeros(population_size)

        for i in range(population_size):
            # Positive perturbation
            theta_plus = theta + sigma * epsilon[i]
            gcn.set_flat_params(theta_plus)
            pred_plus = gcn.forward(X, A)
            losses_plus[i] = binary_cross_entropy(y, pred_plus)

            # Negative perturbation (mirror)
            theta_minus = theta - sigma * epsilon[i]
            gcn.set_flat_params(theta_minus)
            pred_minus = gcn.forward(X, A)
            losses_minus[i] = binary_cross_entropy(y, pred_minus)

        # NES gradient estimate
        advantages = losses_plus - losses_minus
        grad = np.mean(
            advantages[:, np.newaxis] * epsilon, axis=0
        ) / (2.0 * sigma)

        # Gradient descent on loss
        theta = theta - lr * grad

        # Evaluate current loss
        gcn.set_flat_params(theta)
        pred = gcn.forward(X, A)
        current_loss = binary_cross_entropy(y, pred)
        loss_history.append(current_loss)

        # Adaptive noise decay
        if epoch > 0 and epoch % 20 == 0:
            sigma = max(sigma * 0.9, 0.01)

    return loss_history


def evaluate_gcn(
    gcn: GCN,
    X: NDArray[np.float64],
    A: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict:
    """Evaluate GCN performance on binary classification.

    Parameters
    ----------
    gcn : GCN
        Trained graph convolutional network.
    X : ndarray of shape (n_nodes, n_features)
        Node feature matrix.
    A : ndarray of shape (n_nodes, n_nodes)
        Adjacency matrix.
    y : ndarray of shape (n_nodes,)
        True binary labels.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, auc.
    """
    y_pred_prob = gcn.forward(X, A).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(float)
    y_true = y.ravel()

    # Accuracy
    accuracy = float(np.mean(y_pred == y_true))

    # Precision, recall, F1
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # AUC: simple trapezoidal approximation
    auc = _compute_auc(y_true, y_pred_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def _compute_auc(
    y_true: NDArray[np.float64],
    y_scores: NDArray[np.float64],
) -> float:
    """Compute AUC-ROC using the trapezoidal rule.

    Parameters
    ----------
    y_true : ndarray of shape (n,)
        True binary labels.
    y_scores : ndarray of shape (n,)
        Predicted scores/probabilities.

    Returns
    -------
    float
        AUC value in [0, 1]. Returns 0.5 if only one class present.
    """
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by descending score
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]

    # Compute TPR and FPR at each threshold
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0.0
    fp = 0.0

    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal AUC
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    auc = float(np.trapz(tpr_arr, fpr_arr))

    return auc
