"""Macro-to-portfolio transmission models.

Implements OLS-based macro factor regression for translating scenario shocks
into portfolio loss estimates, and credit migration utilities for estimating
losses under stressed transition matrices.

References:
    - Basel Committee (2018), Stress Testing Principles, SS 6--8.
    - Federal Reserve SR 12-7, Supervisory Guidance on Stress Testing.
    - Jarrow, Lando & Turnbull (1997), A Markov model for the term
      structure of credit risk spreads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


class MacroTransmissionModel:
    """OLS regression model mapping macro factor changes to portfolio returns.

    After fitting, the model stores factor betas, residual standard
    deviation, and R-squared.  It can then predict single-period or
    multi-period cumulative losses under a given macro scenario.
    """

    def __init__(self) -> None:
        self.betas: np.ndarray | None = None
        self.intercept: float = 0.0
        self.r_squared: float = 0.0
        self.residual_std: float = 0.0
        self.factor_names: list[str] = []
        self._t_stats: np.ndarray | None = None
        self._p_values: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        portfolio_returns: np.ndarray,
        macro_factors: pd.DataFrame,
    ) -> None:
        """Fit OLS: portfolio_return ~ intercept + beta @ macro_factors.

        Parameters
        ----------
        portfolio_returns : np.ndarray
            1-D array of portfolio returns (length *T*).
        macro_factors : pd.DataFrame
            DataFrame of shape (*T*, *k*) with one column per macro factor.
        """
        y = np.asarray(portfolio_returns, dtype=np.float64).ravel()
        X = macro_factors.values.astype(np.float64)
        n, k = X.shape
        self.factor_names = list(macro_factors.columns)

        # Add intercept column
        X_aug = np.column_stack([np.ones(n), X])

        # OLS: beta_hat = (X'X)^{-1} X'y
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
        beta_hat = XtX_inv @ (X_aug.T @ y)

        self.intercept = float(beta_hat[0])
        self.betas = beta_hat[1:]

        # Residuals and R-squared
        y_hat = X_aug @ beta_hat
        residuals = y - y_hat
        ss_res = float(residuals @ residuals)
        ss_tot = float((y - y.mean()) @ (y - y.mean()))
        self.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        self.residual_std = float(np.sqrt(ss_res / max(n - k - 1, 1)))

        # t-statistics and p-values
        sigma2 = ss_res / max(n - k - 1, 1)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t_stats = beta_hat / se
        dof = max(n - k - 1, 1)
        p_values = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_stats), df=dof))

        # Store stats for factor betas only (skip intercept)
        self._t_stats = t_stats[1:]
        self._p_values = p_values[1:]
        self._fitted = True

    # --------------------------------------------------------- predict_loss
    def predict_loss(self, macro_shocks: dict[str, float]) -> float:
        """Predict portfolio loss from a single-period shock vector.

        loss = -sum(beta_i * shock_i)

        A *negative* portfolio return (caused by adverse shocks) maps to a
        *positive* loss.

        Parameters
        ----------
        macro_shocks : dict[str, float]
            Mapping of factor name -> shock magnitude.

        Returns
        -------
        float
            Predicted portfolio loss.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        shock_vec = np.array(
            [macro_shocks.get(f, 0.0) for f in self.factor_names],
            dtype=np.float64,
        )
        predicted_return = self.intercept + float(self.betas @ shock_vec)
        return -predicted_return

    # ----------------------------------------------------- sensitivity_table
    def sensitivity_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising factor sensitivities.

        Returns
        -------
        pd.DataFrame
            Columns: factor, beta, t_stat, p_value, contribution_pct.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before querying sensitivity.")
        abs_betas = np.abs(self.betas)
        total = abs_betas.sum() if abs_betas.sum() > 0 else 1.0
        return pd.DataFrame({
            "factor": self.factor_names,
            "beta": self.betas,
            "t_stat": self._t_stats,
            "p_value": self._p_values,
            "contribution_pct": abs_betas / total * 100.0,
        })

    # --------------------------------------------------------- predict_path
    def predict_path(self, scenario_df: pd.DataFrame) -> pd.Series:
        """Predict quarter-by-quarter cumulative loss under a multi-period scenario.

        Parameters
        ----------
        scenario_df : pd.DataFrame
            DataFrame indexed by quarter with columns matching the fitted
            factor names.

        Returns
        -------
        pd.Series
            Cumulative loss at each quarter, indexed by the scenario index.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        quarterly_losses: list[float] = []
        for _, row in scenario_df.iterrows():
            shocks = {f: row[f] for f in self.factor_names if f in row.index}
            quarterly_losses.append(self.predict_loss(shocks))
        cumulative = np.cumsum(quarterly_losses)
        return pd.Series(cumulative, index=scenario_df.index, name="cumulative_loss")


# ---------------------------------------------------------------------------
# Credit migration utilities
# ---------------------------------------------------------------------------

def stress_transition_matrix(
    base_matrix: np.ndarray,
    stress_factor: float,
) -> np.ndarray:
    """Stress a credit transition matrix by amplifying downgrade probabilities.

    Off-diagonal elements below the main diagonal (downgrades) are multiplied
    by *stress_factor*.  Each row is then renormalised to sum to 1.

    Parameters
    ----------
    base_matrix : np.ndarray
        Square transition matrix of shape (*n*, *n*) where element (i, j) is
        the probability of migrating from rating *i* to rating *j*.
        Rows sum to 1.  Lower-triangular off-diagonals are downgrades.
    stress_factor : float
        Multiplicative factor applied to downgrade probabilities (> 1
        increases severity).

    Returns
    -------
    np.ndarray
        Stressed transition matrix with rows summing to 1.
    """
    n = base_matrix.shape[0]
    stressed = base_matrix.copy().astype(np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            # j > i means a downgrade (higher index = worse rating)
            stressed[i, j] *= stress_factor

        # Renormalise row
        row_sum = stressed[i].sum()
        if row_sum > 0:
            stressed[i] /= row_sum

    return stressed


def portfolio_loss_under_migration(
    exposures: np.ndarray,
    ratings: np.ndarray,
    transition_matrix: np.ndarray,
    lgd: float,
) -> float:
    """Compute expected loss from credit migration to the default state.

    For each obligor, the expected loss is:
        EL_i = exposure_i * P(default | current_rating_i) * LGD

    The default state is the last column of the transition matrix.

    Parameters
    ----------
    exposures : np.ndarray
        1-D array of exposure amounts for each obligor.
    ratings : np.ndarray
        1-D integer array of current rating indices (0-based).
    transition_matrix : np.ndarray
        Square transition matrix; last column = default probability.
    lgd : float
        Loss given default (fraction, e.g. 0.45).

    Returns
    -------
    float
        Total expected loss across the portfolio.
    """
    default_col = transition_matrix.shape[1] - 1
    total_loss = 0.0
    for i in range(len(exposures)):
        pd_i = transition_matrix[int(ratings[i]), default_col]
        total_loss += exposures[i] * pd_i * lgd
    return float(total_loss)
