"""Policy network for RL portfolio allocation in pure numpy.

Implements a feedforward neural network with ReLU hidden activations and
softmax output layer, ensuring portfolio weights are non-negative and sum
to 1 (valid simplex constraint).

No autograd -- parameter updates are handled externally via NES
(see trainer.py).

Reference: Based on P08 NeuralNetwork architecture with softmax output
for portfolio weight generation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PolicyNetwork:
    """Feedforward neural network with ReLU activations and softmax output.

    Architecture: input -> hidden1 (ReLU) -> hidden2 (ReLU) -> output (softmax)

    The softmax ensures valid portfolio weights:
        w_i = exp(z_i) / sum(exp(z_j)),  all in [0,1], sum=1

    Parameters
    ----------
    state_dim : int
        Dimensionality of the input state vector.
    action_dim : int
        Number of assets (output dimensionality).
    hidden_dim : int
        Width of each hidden layer.
    seed : int
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self._weights: list[NDArray[np.float64]] = []
        self._biases: list[NDArray[np.float64]] = []

        # Layer sizes: [state_dim, hidden_dim, hidden_dim, action_dim]
        layer_sizes = [state_dim, hidden_dim, hidden_dim, action_dim]
        self.n_layers = len(layer_sizes) - 1

        rng = np.random.default_rng(seed)

        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier (Glorot) initialisation
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.standard_normal((fan_in, fan_out)) * scale
            b = np.zeros(fan_out)
            self._weights.append(W.astype(np.float64))
            self._biases.append(b.astype(np.float64))

    def forward(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass: state -> portfolio weights via softmax.

        Parameters
        ----------
        state : ndarray of shape (state_dim,) or (batch_size, state_dim)
            Input state vector(s).

        Returns
        -------
        weights : ndarray of shape (action_dim,) or (batch_size, action_dim)
            Portfolio weights in the simplex (non-negative, sum to 1).
        """
        # Handle 1-D input
        squeeze = False
        if state.ndim == 1:
            state = state.reshape(1, -1)
            squeeze = True

        h = state.astype(np.float64)

        # Hidden layers with ReLU
        for i in range(self.n_layers - 1):
            h = h @ self._weights[i] + self._biases[i]
            h = np.maximum(h, 0.0)  # ReLU

        # Output layer (logits)
        logits = h @ self._weights[-1] + self._biases[-1]

        # Softmax for valid portfolio weights
        weights = _softmax(logits)

        if squeeze:
            return weights.ravel()
        return weights

    def get_flat_params(self) -> NDArray[np.float64]:
        """Flatten all parameters into a single 1-D vector.

        Returns
        -------
        flat : ndarray of shape (num_params,)
        """
        parts = []
        for i in range(self.n_layers):
            parts.append(self._weights[i].ravel())
            parts.append(self._biases[i].ravel())
        return np.concatenate(parts)

    def set_flat_params(self, params: NDArray[np.float64]) -> None:
        """Set parameters from a flattened 1-D vector.

        Parameters
        ----------
        params : ndarray of shape (num_params,)
        """
        idx = 0
        for i in range(self.n_layers):
            w_shape = self._weights[i].shape
            w_size = self._weights[i].size
            self._weights[i] = params[idx : idx + w_size].reshape(w_shape).copy()
            idx += w_size

            b_shape = self._biases[i].shape
            b_size = self._biases[i].size
            self._biases[i] = params[idx : idx + b_size].reshape(b_shape).copy()
            idx += b_size

    @property
    def num_params(self) -> int:
        """Total number of scalar parameters in the network."""
        count = 0
        for i in range(self.n_layers):
            count += self._weights[i].size + self._biases[i].size
        return count


def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax.

    Parameters
    ----------
    x : ndarray of shape (batch_size, n)
        Raw logits.

    Returns
    -------
    probs : ndarray of shape (batch_size, n)
        Probabilities summing to 1 along last axis.
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
