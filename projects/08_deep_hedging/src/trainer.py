"""Training loop for deep hedging using evolutionary strategies.

Since we implement the neural network in pure numpy (no autograd), we use
a simplified CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for
parameter optimisation. This is a derivative-free approach that maintains
a population of candidate solutions and adapts toward lower-risk regions
of parameter space.

Reference:
- Hansen (2016), "The CMA Evolution Strategy: A Tutorial"
- Buehler et al. (2019), "Deep hedging", Quantitative Finance, 19(8).
"""

from __future__ import annotations

import numpy as np
from environment import HedgingEnvironment
from network import NeuralNetwork
from numpy.typing import NDArray


class DeepHedgingTrainer:
    """Trains a neural network to hedge a European call option.

    Uses a simplified evolutionary strategy (ES) to optimise network
    parameters, minimising a risk measure of the hedging P&L distribution.

    Parameters
    ----------
    env : HedgingEnvironment
        The hedging simulation environment.
    network : NeuralNetwork
        The neural network mapping market state to hedge ratios.
    config : dict
        Training configuration with keys:
        - population_size: int, number of candidate perturbations per epoch
        - lr: float, learning rate for parameter updates
        - risk_measure: str, one of "variance", "cvar", "mean_variance"
        - cvar_alpha: float, quantile level for CVaR (default 0.95)
        - lambda_mv: float, weight on variance in mean_variance objective
    """

    def __init__(
        self,
        env: HedgingEnvironment,
        network: NeuralNetwork,
        config: dict,
    ) -> None:
        self.env = env
        self.network = network
        self.population_size = config.get("population_size", 50)
        self.lr = config.get("lr", 0.02)
        self.measure_name = config.get("risk_measure", "cvar")
        self.cvar_alpha = config.get("cvar_alpha", 0.95)
        self.lambda_mv = config.get("lambda_mv", 1.0)
        self._noise_std = config.get("noise_std", 0.1)

    def compute_hedge_positions(
        self, paths: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Run the network at each time step to produce hedge positions.

        At each step t, the network receives features:
            [S_t / S_0  (normalised price),
             t / T       (time fraction),
             current_position  (previous hedge ratio, 0 at t=0)]

        Parameters
        ----------
        paths : ndarray of shape (n_paths, n_steps + 1)
            Simulated price paths.

        Returns
        -------
        positions : ndarray of shape (n_paths, n_steps)
            Hedge positions (delta) at each rebalancing date.
        """
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1
        positions = np.zeros((n_paths, n_steps))

        current_pos = np.zeros(n_paths)

        for t in range(n_steps):
            # Feature vector: [S_t/S_0, t/T, current_position]
            s_norm = paths[:, t] / self.env.s0
            t_frac = np.full(n_paths, t * self.env.dt / self.env.T)
            features = np.column_stack([s_norm, t_frac, current_pos])

            # Network output: delta in [0, 1]
            delta = self.network.forward(features).ravel()
            positions[:, t] = delta
            current_pos = delta

        return positions

    def risk_measure(self, pnl: NDArray[np.float64], measure: str) -> float:
        """Compute a risk measure of the P&L distribution.

        We negate the P&L because risk measures operate on losses.
        The objective is to minimise risk, so we want to minimise the
        risk measure of the loss distribution.

        Parameters
        ----------
        pnl : ndarray of shape (n_paths,)
            Hedging P&L (positive = profit).
        measure : str
            One of "variance", "cvar", "mean_variance".

        Returns
        -------
        risk : float
            Risk measure value (lower is better hedge).
        """
        losses = -pnl  # convert profit to loss

        if measure == "variance":
            return float(np.var(losses))
        elif measure == "cvar":
            alpha = self.cvar_alpha
            var_level = np.percentile(losses, alpha * 100)
            tail = losses[losses >= var_level]
            if len(tail) == 0:
                return float(var_level)
            return float(np.mean(tail))
        elif measure == "mean_variance":
            return float(np.mean(losses) + self.lambda_mv * np.var(losses))
        else:
            raise ValueError(
                f"Unknown risk measure '{measure}'. "
                "Choose from 'variance', 'cvar', 'mean_variance'."
            )

    def _evaluate(
        self, flat_params: NDArray[np.float64], paths: NDArray[np.float64]
    ) -> float:
        """Evaluate risk for a given parameter vector.

        Parameters
        ----------
        flat_params : ndarray
            Flattened network parameters.
        paths : ndarray
            Simulated price paths.

        Returns
        -------
        risk : float
            Risk measure value.
        """
        self.network.set_flat_parameters(flat_params)
        positions = self.compute_hedge_positions(paths)
        pnl = self.env.compute_pnl(paths, positions)
        return self.risk_measure(pnl, self.measure_name)

    def train(
        self,
        n_epochs: int = 100,
        lr: float | None = None,
        measure: str | None = None,
    ) -> list[float]:
        """Train the network using Natural Evolution Strategies (NES).

        At each epoch:
        1. Sample a population of perturbation vectors.
        2. Evaluate the risk for each perturbed parameter vector.
        3. Estimate the gradient using the NES formula.
        4. Update parameters in the direction of lower risk.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        lr : float or None
            Learning rate override. If None, uses the value from config.
        measure : str or None
            Risk measure override. If None, uses the value from config.

        Returns
        -------
        loss_history : list of float
            Risk measure value at each epoch.
        """
        if lr is not None:
            self.lr = lr
        if measure is not None:
            self.measure_name = measure

        # Simulate training paths once (fixed for all epochs)
        paths = self.env.simulate_paths()

        # Current best parameters
        theta = self.network.get_flat_parameters()
        n_params = len(theta)

        loss_history: list[float] = []
        sigma = self._noise_std  # perturbation scale

        rng = np.random.default_rng(self.env.seed)

        for epoch in range(n_epochs):
            # Sample perturbations: (population_size, n_params)
            epsilon = rng.standard_normal((self.population_size, n_params))

            # Evaluate risk for each perturbation (mirror sampling for variance reduction)
            risks_plus = np.zeros(self.population_size)
            risks_minus = np.zeros(self.population_size)

            for i in range(self.population_size):
                theta_plus = theta + sigma * epsilon[i]
                theta_minus = theta - sigma * epsilon[i]
                risks_plus[i] = self._evaluate(theta_plus, paths)
                risks_minus[i] = self._evaluate(theta_minus, paths)

            # NES gradient estimate with mirror sampling:
            # grad ~= (1 / (2 * pop_size * sigma)) * sum((f+ - f-) * epsilon)
            advantages = risks_plus - risks_minus  # shape (pop_size,)
            grad = np.mean(
                advantages[:, np.newaxis] * epsilon, axis=0
            ) / (2.0 * sigma)

            # Update parameters (gradient descent on risk)
            theta = theta - self.lr * grad

            # Evaluate current risk
            current_risk = self._evaluate(theta, paths)
            loss_history.append(current_risk)

            # Restore best parameters
            self.network.set_flat_parameters(theta)

            # Adaptive noise decay
            if epoch > 0 and epoch % 20 == 0:
                sigma = max(sigma * 0.9, 0.01)

        return loss_history
