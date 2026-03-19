"""Graph Convolutional Network in pure numpy.

Implements a two-layer GCN following Kipf & Welling (2017):
    H^{(l+1)} = sigma( D_hat^{-1/2} A_hat D_hat^{-1/2} H^{(l)} W^{(l)} + b^{(l)} )

where A_hat = A + I (adjacency with self-loops), D_hat = diag(A_hat 1),
and sigma is a nonlinearity (ReLU for hidden, sigmoid for output).

No autograd -- parameter optimisation is done externally via evolutionary
strategies (see trainer.py).

Reference:
- Kipf & Welling (2017), "Semi-supervised classification with graph
  convolutional networks", ICLR.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """ReLU activation."""
    return np.maximum(x, 0.0)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid activation."""
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class GCN:
    """Two-layer Graph Convolutional Network.

    Architecture:
        Layer 1: A_norm @ X @ W1 + b1, followed by ReLU
        Layer 2: A_norm @ H1 @ W2 + b2, followed by sigmoid

    Output shape: (n_nodes, n_classes), values in [0, 1].

    Parameters
    ----------
    n_features : int
        Number of input features per node.
    hidden_dim : int
        Number of hidden units in the first GCN layer.
    n_classes : int
        Number of output classes (1 for binary classification).
    seed : int
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_classes: int,
        seed: int,
    ) -> None:
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        rng = np.random.default_rng(seed)

        # Xavier initialisation for layer 1: (n_features, hidden_dim)
        limit1 = np.sqrt(6.0 / (n_features + hidden_dim))
        self.W1 = rng.uniform(-limit1, limit1, size=(n_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # Xavier initialisation for layer 2: (hidden_dim, n_classes)
        limit2 = np.sqrt(6.0 / (hidden_dim + n_classes))
        self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, n_classes))
        self.b2 = np.zeros(n_classes)

    def forward(
        self,
        X: NDArray[np.float64],
        A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Forward pass through the two-layer GCN.

        Parameters
        ----------
        X : ndarray of shape (n_nodes, n_features)
            Node feature matrix.
        A : ndarray of shape (n_nodes, n_nodes)
            Adjacency matrix (weighted or binary).

        Returns
        -------
        ndarray of shape (n_nodes, n_classes)
            Predicted probabilities in [0, 1].
        """
        n = A.shape[0]

        # Step 1: A_hat = A + I (add self-loops)
        A_hat = A + np.eye(n)

        # Step 2: D = diag(A_hat.sum(axis=1))
        d = A_hat.sum(axis=1)

        # Step 3: D_inv_sqrt = D^{-1/2}
        d_inv_sqrt = np.zeros(n)
        nonzero = d > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)

        # Step 4: A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        # Step 5: H1 = relu(A_norm @ X @ W1 + b1)
        H1 = _relu(A_norm @ X @ self.W1 + self.b1)

        # Step 6: H2 = sigmoid(A_norm @ H1 @ W2 + b2)
        H2 = _sigmoid(A_norm @ H1 @ self.W2 + self.b2)

        return H2

    def get_flat_params(self) -> NDArray[np.float64]:
        """Flatten all parameters into a single 1D array.

        Returns
        -------
        ndarray
            Concatenated [W1.flat, b1, W2.flat, b2].
        """
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel(),
        ])

    def set_flat_params(self, params: NDArray[np.float64]) -> None:
        """Restore parameters from a flat 1D array.

        Parameters
        ----------
        params : ndarray
            Flat parameter vector (same size as get_flat_params output).
        """
        idx = 0

        size_W1 = self.n_features * self.hidden_dim
        self.W1 = params[idx:idx + size_W1].reshape(
            self.n_features, self.hidden_dim
        )
        idx += size_W1

        size_b1 = self.hidden_dim
        self.b1 = params[idx:idx + size_b1].copy()
        idx += size_b1

        size_W2 = self.hidden_dim * self.n_classes
        self.W2 = params[idx:idx + size_W2].reshape(
            self.hidden_dim, self.n_classes
        )
        idx += size_W2

        size_b2 = self.n_classes
        self.b2 = params[idx:idx + size_b2].copy()
        idx += size_b2

    def num_params(self) -> int:
        """Total number of trainable parameters.

        Returns
        -------
        int
        """
        return (
            self.n_features * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.n_classes
            + self.n_classes
        )
