"""Simple feedforward neural network in pure numpy.

Implements a fully-connected network with ReLU hidden activations and
sigmoid output layer, suitable for learning hedge ratios in [0, 1].

No autograd -- parameter updates are handled externally via evolutionary
strategies (see trainer.py).

Reference: Buehler et al. (2019), Section 3 -- Network architecture.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class NeuralNetwork:
    """Feedforward neural network with ReLU activations and sigmoid output.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer including input and output.
        E.g. [3, 32, 32, 1] for 3 inputs, two hidden layers of 32, 1 output.
    seed : int
        Random seed for weight initialisation.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 42) -> None:
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self._weights: list[NDArray[np.float64]] = []
        self._biases: list[NDArray[np.float64]] = []

        rng = np.random.default_rng(seed)

        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier (Glorot) initialisation: scale = sqrt(2 / (fan_in + fan_out))
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.standard_normal((fan_in, fan_out)) * scale
            b = np.zeros(fan_out)
            self._weights.append(W.astype(np.float64))
            self._biases.append(b.astype(np.float64))

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass through the network.

        Hidden layers use ReLU activation. Output layer uses sigmoid
        to bound the hedge ratio in [0, 1] (appropriate for a long call).

        Parameters
        ----------
        x : ndarray of shape (batch_size, input_dim)
            Input features.

        Returns
        -------
        output : ndarray of shape (batch_size, output_dim)
            Network output in [0, 1].
        """
        h = x.astype(np.float64)

        for i in range(self.n_layers - 1):
            h = h @ self._weights[i] + self._biases[i]
            # ReLU activation
            h = np.maximum(h, 0.0)

        # Output layer with sigmoid
        h = h @ self._weights[-1] + self._biases[-1]
        output = _sigmoid(h)

        return output

    def parameters(self) -> list[NDArray[np.float64]]:
        """Return a flat list of all weight and bias arrays.

        Returns
        -------
        params : list of ndarray
            [W0, b0, W1, b1, ...] in layer order.
        """
        params: list[NDArray[np.float64]] = []
        for i in range(self.n_layers):
            params.append(self._weights[i].copy())
            params.append(self._biases[i].copy())
        return params

    def set_parameters(self, params: list[NDArray[np.float64]]) -> None:
        """Set network parameters from a flat list.

        Parameters
        ----------
        params : list of ndarray
            Must match the structure from ``parameters()``.
        """
        for i in range(self.n_layers):
            self._weights[i] = params[2 * i].copy()
            self._biases[i] = params[2 * i + 1].copy()

    def num_parameters(self) -> int:
        """Total number of scalar parameters in the network.

        Returns
        -------
        count : int
        """
        count = 0
        for i in range(self.n_layers):
            count += self._weights[i].size + self._biases[i].size
        return count

    def get_flat_parameters(self) -> NDArray[np.float64]:
        """Flatten all parameters into a single 1-D vector.

        Returns
        -------
        flat : ndarray of shape (num_parameters,)
        """
        parts = []
        for p in self.parameters():
            parts.append(p.ravel())
        return np.concatenate(parts)

    def set_flat_parameters(self, flat: NDArray[np.float64]) -> None:
        """Set parameters from a flattened 1-D vector.

        Parameters
        ----------
        flat : ndarray of shape (num_parameters,)
        """
        idx = 0
        params: list[NDArray[np.float64]] = []
        for i in range(self.n_layers):
            w_shape = self._weights[i].shape
            w_size = self._weights[i].size
            params.append(flat[idx : idx + w_size].reshape(w_shape))
            idx += w_size

            b_shape = self._biases[i].shape
            b_size = self._biases[i].size
            params.append(flat[idx : idx + b_size].reshape(b_shape))
            idx += b_size
        self.set_parameters(params)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid function."""
    # Clip to avoid overflow in exp
    x = np.clip(x, -500.0, 500.0)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
