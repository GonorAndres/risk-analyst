"""Hedging environment for deep hedging.

Simulates GBM paths and provides Black-Scholes benchmark computations
for European call option hedging.

Reference: Buehler et al. (2019), "Deep hedging", Quantitative Finance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


class HedgingEnvironment:
    """Simulation environment for hedging a European call option.

    Wraps GBM path simulation with Black-Scholes analytics for benchmarking
    neural-network-learned hedging strategies.

    Parameters
    ----------
    s0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualised, continuously compounded).
    sigma : float
        Volatility (annualised).
    T : float
        Time to maturity in years.
    n_steps : int
        Number of hedging rebalancing dates.
    n_paths : int
        Number of Monte Carlo paths.
    cost_rate : float
        Proportional transaction cost rate (e.g. 0.001 = 10 bps).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        s0: float = 100.0,
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.20,
        T: float = 1 / 12,
        n_steps: int = 21,
        n_paths: int = 5000,
        cost_rate: float = 0.001,
        seed: int | None = 42,
    ) -> None:
        self.s0 = s0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.cost_rate = cost_rate
        self.seed = seed
        self.dt = T / n_steps

    def simulate_paths(self) -> NDArray[np.float64]:
        """Simulate GBM paths under the risk-neutral measure.

        Uses the exact log-normal solution:
            S(t+dt) = S(t) * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

        Returns
        -------
        paths : ndarray of shape (n_paths, n_steps + 1)
            Simulated price paths. paths[:, 0] == s0.
        """
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal((self.n_paths, self.n_steps))

        log_increments = (
            (self.r - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )

        log_paths = np.zeros((self.n_paths, self.n_steps + 1))
        log_paths[:, 0] = np.log(self.s0)
        log_paths[:, 1:] = np.log(self.s0) + np.cumsum(log_increments, axis=1)

        return np.exp(log_paths)

    def bs_delta(
        self, S: float | NDArray[np.float64], t: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Black-Scholes delta for a European call.

        delta = N(d1) where d1 = [ln(S/K) + (r + sigma^2/2)(T-t)] / [sigma*sqrt(T-t)]

        Parameters
        ----------
        S : float or array
            Current stock price(s).
        t : float or array
            Current time(s) (0 <= t < T).

        Returns
        -------
        delta : ndarray
            Black-Scholes delta, in [0, 1].
        """
        S = np.asarray(S, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        tau = np.maximum(self.T - t, 1e-10)  # time to maturity

        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        return norm.cdf(d1)

    def bs_price(
        self, S: float | NDArray[np.float64], t: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Black-Scholes price for a European call.

        C = S*N(d1) - K*exp(-r*(T-t))*N(d2)

        Parameters
        ----------
        S : float or array
            Current stock price(s).
        t : float or array
            Current time(s).

        Returns
        -------
        price : ndarray
            Black-Scholes call price.
        """
        S = np.asarray(S, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        tau = np.maximum(self.T - t, 1e-10)

        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        d2 = d1 - self.sigma * np.sqrt(tau)

        price = S * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
        return np.asarray(price, dtype=np.float64)

    def compute_payoff(self, S_T: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute European call payoff: max(S_T - K, 0).

        Parameters
        ----------
        S_T : ndarray
            Terminal stock prices.

        Returns
        -------
        payoff : ndarray
            Call payoffs (non-negative).
        """
        return np.maximum(S_T - self.K, 0.0)

    def compute_pnl(
        self,
        paths: NDArray[np.float64],
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hedging P&L for each path.

        P&L = option_premium - payoff + hedging_gains - transaction_costs

        The hedge P&L from the writer's perspective:
        - Receive the option premium at t=0
        - Pay the payoff at maturity
        - Hedging gains/losses from delta positions
        - Transaction costs from rebalancing

        Parameters
        ----------
        paths : ndarray of shape (n_paths, n_steps + 1)
            Simulated price paths.
        positions : ndarray of shape (n_paths, n_steps)
            Hedge positions (number of shares) at each step.

        Returns
        -------
        pnl : ndarray of shape (n_paths,)
            Hedging P&L for each path (from writer's perspective).
            Positive = profit, Negative = loss.
        """
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        # Option premium received at t=0 (BS price)
        premium = float(self.bs_price(self.s0, 0.0))

        # Payoff at maturity
        payoff = self.compute_payoff(paths[:, -1])

        # Hedging gains: sum over steps of position * price_change
        # At step t, we hold positions[:, t] shares, and price moves from
        # paths[:, t] to paths[:, t+1]
        price_changes = np.diff(paths, axis=1)  # (n_paths, n_steps)
        hedging_gains = np.sum(positions * price_changes, axis=1)

        # Transaction costs: proportional to |change in position| * price
        # Initial trade: buy positions[:, 0] shares at paths[:, 0]
        costs = np.zeros(n_paths)
        costs += self.cost_rate * np.abs(positions[:, 0]) * paths[:, 0]

        # Subsequent rebalancing
        for t in range(1, n_steps):
            delta_pos = np.abs(positions[:, t] - positions[:, t - 1])
            costs += self.cost_rate * delta_pos * paths[:, t]

        # Final unwinding at maturity
        costs += self.cost_rate * np.abs(positions[:, -1]) * paths[:, -1]

        # P&L from writer's perspective
        pnl = premium - payoff + hedging_gains - costs

        return pnl
