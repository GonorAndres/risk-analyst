"""Portfolio MDP environment for reinforcement learning.

Implements a gym-style environment where an RL agent dynamically allocates
capital across multiple assets, subject to transaction costs and a
CVaR-based risk penalty.

State vector: [flattened recent returns (lookback * n_assets),
               current_weights (n_assets),
               rolling_vol,
               current_drawdown]

Action: portfolio weight vector in the simplex (non-negative, sums to 1).

Reference: Wang et al. (2025), ICVaR-DRL for portfolio optimization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PortfolioEnv:
    """Markov Decision Process for portfolio allocation.

    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio.
    lookback : int
        Number of past return observations in the state.
    initial_capital : float
        Starting portfolio value.
    cost_rate : float
        Proportional transaction cost rate.
    cvar_alpha : float
        Confidence level for CVaR penalty (e.g. 0.95).
    cvar_threshold : float
        CVaR threshold below which no penalty is applied.
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int = 10,
        initial_capital: float = 1.0,
        cost_rate: float = 0.001,
        cvar_alpha: float = 0.95,
        cvar_threshold: float = 0.03,
    ) -> None:
        self.n_assets = n_assets
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.cost_rate = cost_rate
        self.cvar_alpha = cvar_alpha
        self.cvar_threshold = cvar_threshold

        # Internal state
        self._returns_data: NDArray[np.float64] | None = None
        self._t: int = 0
        self._capital: float = initial_capital
        self._peak_capital: float = initial_capital
        self._weights: NDArray[np.float64] = np.zeros(n_assets)
        self._portfolio_returns: list[float] = []

    @property
    def state_dim(self) -> int:
        """Total dimensionality of the state vector."""
        return self.lookback * self.n_assets + self.n_assets + 2

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action (portfolio weights)."""
        return self.n_assets

    def reset(self, returns_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reset environment with new return data.

        Parameters
        ----------
        returns_data : ndarray of shape (T, n_assets)
            Matrix of asset returns for the episode.

        Returns
        -------
        state : ndarray of shape (state_dim,)
            Initial state vector.
        """
        self._returns_data = returns_data.astype(np.float64)
        self._t = self.lookback  # start after enough history
        self._capital = self.initial_capital
        self._peak_capital = self.initial_capital
        self._weights = np.ones(self.n_assets) / self.n_assets  # equal weight start
        self._portfolio_returns = []
        return self.get_state()

    def step(
        self, weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, dict]:
        """Execute one time step of the MDP.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Target portfolio weights (should sum to 1, non-negative).

        Returns
        -------
        next_state : ndarray of shape (state_dim,)
        reward : float
        done : bool
        info : dict
            Contains portfolio_return, cost, capital, drawdown.
        """
        assert self._returns_data is not None, "Call reset() before step()"

        # Asset returns at current time step
        asset_returns = self._returns_data[self._t]

        # Portfolio return before costs
        portfolio_return = float(np.dot(weights, asset_returns))

        # Transaction cost: proportional to weight changes
        cost = self.cost_rate * float(np.sum(np.abs(weights - self._weights)))

        # Net return
        net_return = portfolio_return - cost

        # Update capital
        self._capital *= (1.0 + net_return)
        self._peak_capital = max(self._peak_capital, self._capital)

        # Store return and update weights
        self._portfolio_returns.append(portfolio_return)
        self._weights = weights.copy()
        self._t += 1

        # Check if episode is done
        done = self._t >= len(self._returns_data)

        # Compute reward
        reward = self._compute_reward(portfolio_return, cost)

        # Next state
        next_state = self.get_state() if not done else np.zeros(self.state_dim)

        info = {
            "portfolio_return": portfolio_return,
            "cost": cost,
            "capital": self._capital,
            "drawdown": self._current_drawdown(),
        }

        return next_state, reward, done, info

    def get_state(self) -> NDArray[np.float64]:
        """Construct the current state vector.

        State = [flattened recent returns (lookback * n_assets),
                 current_weights (n_assets),
                 rolling_vol,
                 current_drawdown]

        Returns
        -------
        state : ndarray of shape (state_dim,)
        """
        assert self._returns_data is not None

        # Recent returns window
        start = max(0, self._t - self.lookback)
        recent = self._returns_data[start : self._t]

        # Pad if not enough history
        if len(recent) < self.lookback:
            pad = np.zeros((self.lookback - len(recent), self.n_assets))
            recent = np.vstack([pad, recent])

        flat_returns = recent.flatten()

        # Rolling volatility of portfolio returns
        if len(self._portfolio_returns) >= 2:
            rolling_vol = float(np.std(self._portfolio_returns[-min(20, len(self._portfolio_returns)):]))
        else:
            rolling_vol = 0.0

        # Current drawdown
        drawdown = self._current_drawdown()

        state = np.concatenate([
            flat_returns,
            self._weights,
            np.array([rolling_vol, drawdown]),
        ])

        return state.astype(np.float64)

    def _current_drawdown(self) -> float:
        """Compute current drawdown from peak capital."""
        if self._peak_capital <= 0:
            return 0.0
        return float((self._peak_capital - self._capital) / self._peak_capital)

    def _compute_reward(self, portfolio_return: float, cost: float) -> float:
        """Compute step reward: return - lambda * cvar_penalty - cost_penalty.

        Parameters
        ----------
        portfolio_return : float
            Raw portfolio return at this step.
        cost : float
            Transaction cost incurred.

        Returns
        -------
        reward : float
        """
        # CVaR penalty: penalise when empirical CVaR exceeds threshold
        cvar_penalty = 0.0
        if len(self._portfolio_returns) >= 10:
            losses = -np.array(self._portfolio_returns)
            sorted_losses = np.sort(losses)
            n = len(sorted_losses)
            cutoff = int(np.ceil(n * self.cvar_alpha))
            if cutoff < n:
                tail = sorted_losses[cutoff:]
                cvar_val = float(np.mean(tail))
                if cvar_val > self.cvar_threshold:
                    cvar_penalty = cvar_val - self.cvar_threshold

        reward = portfolio_return - cvar_penalty - cost
        return reward


def generate_synthetic_returns(
    n_assets: int,
    n_steps: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate correlated synthetic returns with regime change.

    First half: calm regime (low vol, positive drift).
    Second half: crisis regime (high vol, negative drift for risky assets).

    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_steps : int
        Number of time steps.
    seed : int
        Random seed.

    Returns
    -------
    returns : ndarray of shape (n_steps, n_assets)
    """
    rng = np.random.default_rng(seed)

    half = n_steps // 2

    # Correlation matrix
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            rho = 0.3 * np.exp(-0.5 * abs(i - j))
            corr[i, j] = rho
            corr[j, i] = rho

    L = np.linalg.cholesky(corr)

    # Calm regime: positive drift, low vol
    mu_calm = np.linspace(0.0003, 0.00005, n_assets)
    sigma_calm = np.linspace(0.012, 0.001, n_assets)

    z_calm = rng.standard_normal((half, n_assets))
    returns_calm = mu_calm + (z_calm @ L.T) * sigma_calm

    # Crisis regime: negative drift for risky assets, higher vol
    mu_crisis = np.linspace(-0.001, 0.0001, n_assets)
    sigma_crisis = np.linspace(0.025, 0.002, n_assets)

    z_crisis = rng.standard_normal((n_steps - half, n_assets))
    returns_crisis = mu_crisis + (z_crisis @ L.T) * sigma_crisis

    returns = np.vstack([returns_calm, returns_crisis])
    return returns.astype(np.float64)
