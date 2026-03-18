"""Simple derivative instruments for CVA calculation.

Implements interest rate swap valuation and Vasicek short-rate
simulation.  All pricing is from scratch using numpy/scipy -- no
QuantLib dependency.

Reference: Gregory (2020), Ch. 8 -- Exposure Simulation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_rate_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate short-rate paths under the Vasicek model.

    dr = kappa * (theta - r) * dt + sigma * dW

    Uses Euler-Maruyama discretization.

    Parameters
    ----------
    r0 : float
        Initial short rate.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean level.
    sigma : float
        Volatility of the short rate.
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    paths : ndarray of shape (n_paths, n_steps + 1)
        Simulated short-rate paths.  paths[:, 0] == r0.
    times : ndarray of shape (n_steps + 1,)
        Time grid from 0 to T.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    z = rng.standard_normal((n_paths, n_steps))

    for i in range(n_steps):
        r = paths[:, i]
        paths[:, i + 1] = r + kappa * (theta - r) * dt + sigma * np.sqrt(dt) * z[:, i]

    return paths, times


class InterestRateSwap:
    """Plain-vanilla interest rate swap for CVA exposure calculation.

    The receiver swap receives fixed and pays floating.  Mark-to-market
    is computed as:
        V(t) = notional * (fixed_rate - floating_rate(t)) * remaining_annuity(t)

    Parameters
    ----------
    notional : float
        Notional principal.
    fixed_rate : float
        Fixed coupon rate.
    tenor : float
        Swap maturity in years.
    payment_freq : float
        Payment frequency in years (e.g. 0.25 for quarterly).
    seed : int
        Random seed (unused here but kept for API consistency).
    """

    def __init__(
        self,
        notional: float,
        fixed_rate: float,
        tenor: float,
        payment_freq: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.tenor = tenor
        self.payment_freq = payment_freq
        self.seed = seed

    def simulate_values(
        self,
        rate_paths: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute mark-to-market value of the swap at each time for each scenario.

        V(t) = notional * (fixed_rate - r(t)) * remaining_annuity(t)

        where remaining_annuity(t) = sum of payment_freq for remaining
        payment dates after t, discounted at the current short rate.

        Parameters
        ----------
        rate_paths : ndarray of shape (n_paths, n_times)
            Short-rate paths from Vasicek simulation.
        times : ndarray of shape (n_times,)
            Time grid corresponding to rate_paths columns.

        Returns
        -------
        values : ndarray of shape (n_paths, n_times)
            Mark-to-market swap values at each time for each path.
        """
        n_paths, n_times = rate_paths.shape
        values = np.zeros((n_paths, n_times))

        for j in range(n_times):
            t = times[j]
            remaining_time = self.tenor - t
            if remaining_time <= 0:
                continue

            # Number of remaining payment periods
            n_remaining = int(np.ceil(remaining_time / self.payment_freq))
            if n_remaining == 0:
                continue

            # Compute remaining annuity discounted at current short rate
            # Simple approach: sum of discount factors for each remaining
            # payment date
            r = rate_paths[:, j]  # current short rate for each path
            annuity = np.zeros(n_paths)
            for k in range(1, n_remaining + 1):
                tau = k * self.payment_freq
                # Discount factor using current short rate (flat curve approx)
                annuity += self.payment_freq * np.exp(-r * tau)

            # Mark-to-market: receiver swap value
            values[:, j] = self.notional * (self.fixed_rate - r) * annuity

        return values
