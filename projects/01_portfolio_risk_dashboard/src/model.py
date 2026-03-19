"""Risk model encapsulating VaR/ES computation for a portfolio.

The ``RiskModel`` class is the core analytical engine: fit it with return
data and portfolio weights, then query any VaR method or ES at any
confidence level.

References:
    - Jorion (2007), Ch. 11--12: VaR methods (historical, parametric, MC).
    - Acerbi & Tasche (2002): coherence of Expected Shortfall.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from risk_analyst.data.market import compute_losses
from risk_analyst.measures.var import (
    expected_shortfall,
    historical_var,
    monte_carlo_var,
    parametric_var,
)


class RiskModel:
    """Encapsulates VaR and ES computation for a weighted portfolio.

    Usage
    -----
    >>> model = RiskModel(n_sims=10_000, seed=42)
    >>> model.fit(returns, weights)
    >>> model.var(alpha=0.99, method="historical")
    >>> model.es(alpha=0.99)
    """

    def __init__(self, n_sims: int = 10_000, seed: int = 42) -> None:
        self.n_sims = n_sims
        self.seed = seed
        self._losses: np.ndarray | None = None
        self._returns: pd.DataFrame | None = None
        self._weights: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame, weights: np.ndarray | list[float]) -> RiskModel:
        """Fit the model with asset returns and portfolio weights.

        Computes and stores the portfolio loss series L_t = -r_{p,t}.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset-level returns (T x N).
        weights : array-like
            Portfolio weights summing to 1.

        Returns
        -------
        RiskModel
            Self, for method chaining.
        """
        self._returns = returns
        self._weights = np.asarray(weights, dtype=np.float64)
        loss_series = compute_losses(returns, self._weights)
        self._losses = loss_series.values
        self._loss_index = loss_series.index
        return self

    @property
    def losses(self) -> np.ndarray:
        """Portfolio loss array (positive = loss)."""
        if self._losses is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        return self._losses

    @property
    def loss_series(self) -> pd.Series:
        """Portfolio losses as a pandas Series with date index."""
        if self._losses is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        return pd.Series(self._losses, index=self._loss_index, name="portfolio_loss")

    def var(
        self,
        alpha: float = 0.99,
        method: Literal["historical", "parametric", "monte_carlo"] = "historical",
    ) -> float:
        """Compute Value-at-Risk.

        Parameters
        ----------
        alpha : float
            Confidence level (e.g. 0.95, 0.99).
        method : {"historical", "parametric", "monte_carlo"}
            VaR estimation method.

        Returns
        -------
        float
            VaR estimate.
        """
        losses = self.losses
        if method == "historical":
            return historical_var(losses, alpha)
        elif method == "parametric":
            return parametric_var(losses, alpha)
        elif method == "monte_carlo":
            return monte_carlo_var(losses, alpha, n_sims=self.n_sims, seed=self.seed)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def es(self, alpha: float = 0.99) -> float:
        """Compute Expected Shortfall (CVaR).

        ES_alpha = E[L | L >= VaR_alpha]

        Parameters
        ----------
        alpha : float
            Confidence level.

        Returns
        -------
        float
            Expected Shortfall estimate.
        """
        return expected_shortfall(self.losses, alpha)

    def rolling_var(
        self,
        window: int = 252,
        alpha: float = 0.99,
        method: Literal["historical", "parametric", "monte_carlo"] = "historical",
    ) -> pd.Series:
        """Compute rolling VaR over a sliding window.

        For each date t, VaR is estimated from losses in [t-window, t-1].

        Parameters
        ----------
        window : int
            Lookback window in trading days.
        alpha : float
            Confidence level.
        method : {"historical", "parametric", "monte_carlo"}
            VaR estimation method.

        Returns
        -------
        pd.Series
            Rolling VaR series, indexed by date, with NaN for the
            initial warm-up period.
        """
        losses = self.losses
        n = len(losses)
        var_values = np.full(n, np.nan)

        var_func = {
            "historical": lambda l: historical_var(l, alpha),
            "parametric": lambda l: parametric_var(l, alpha),
            "monte_carlo": lambda l: monte_carlo_var(
                l, alpha, n_sims=self.n_sims, seed=self.seed
            ),
        }[method]

        for t in range(window, n):
            window_losses = losses[t - window : t]
            var_values[t] = var_func(window_losses)

        return pd.Series(var_values, index=self._loss_index, name=f"VaR_{method}_{alpha}")
