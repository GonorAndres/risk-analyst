"""CopulaModel: high-level interface for multi-family copula analysis.

Coordinates marginal filtering, copula fitting across families,
tail-dependence comparison, Monte Carlo simulation, and portfolio
VaR computation.

References:
    - Nelsen (2006), *An Introduction to Copulas*.
    - McNeil, Frey & Embrechts (2015), Ch. 7: copula models for risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_analyst.models.copula import (
    clayton_copula_fit,
    copula_sample,
    frank_copula_fit,
    gaussian_copula_fit,
    gumbel_copula_fit,
    pit_transform,
    t_copula_fit,
    tail_dependence,
)

from marginal import filter_marginals, inverse_pit


# Map family names to fitting functions
_FIT_FUNCTIONS = {
    "gaussian": gaussian_copula_fit,
    "t": t_copula_fit,
    "clayton": clayton_copula_fit,
    "gumbel": gumbel_copula_fit,
    "frank": frank_copula_fit,
}


class CopulaModel:
    """High-level copula dependency model.

    Fits multiple copula families to pseudo-uniform data, compares
    tail-dependence coefficients, simulates joint scenarios, and
    computes portfolio VaR under each copula assumption.

    Parameters
    ----------
    config : dict
        Configuration dictionary (loaded from ``configs/default.yaml``).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.families: list[str] = config["copula"]["families"]
        self.n_samples: int = config["simulation"]["n_samples"]
        self.seed: int = config.get("random_seed", 42)
        self._fitted_params: dict[str, dict] = {}

    def fit_all_families(self, u_data: np.ndarray) -> dict[str, dict]:
        """Fit all configured copula families to *u_data*.

        For Archimedean copulas (Clayton, Gumbel, Frank), which are
        bivariate, the first two columns of *u_data* are used.

        Parameters
        ----------
        u_data : np.ndarray
            Pseudo-uniform observations of shape (n, d).

        Returns
        -------
        dict[str, dict]
            Mapping from family name to fitted parameter dictionary.
        """
        results: dict[str, dict] = {}

        for family in self.families:
            fit_fn = _FIT_FUNCTIONS[family]

            if family in ("clayton", "gumbel", "frank"):
                # Archimedean: bivariate only, use first two columns
                data = u_data[:, :2]
            else:
                data = u_data

            if family == "t":
                df_range = tuple(self.config["copula"]["t_df_range"])
                params = fit_fn(data, df_range=df_range)
            else:
                params = fit_fn(data)

            results[family] = params

        self._fitted_params = results
        return results

    def compare_families(self, u_data: np.ndarray) -> pd.DataFrame:
        """Compare fitted copulas by tail dependence.

        Parameters
        ----------
        u_data : np.ndarray
            Pseudo-uniform observations (used for fitting if not already done).

        Returns
        -------
        pd.DataFrame
            Comparison table with columns: ``family``, ``lambda_L``,
            ``lambda_U``, and any family-specific parameters (``df``,
            ``theta``).
        """
        if not self._fitted_params:
            self.fit_all_families(u_data)

        rows = []
        for family, params in self._fitted_params.items():
            td = tail_dependence(params)
            row = {
                "family": family,
                "lambda_L": td["lambda_L"],
                "lambda_U": td["lambda_U"],
            }
            # Add family-specific params for display
            if "df" in params:
                row["df"] = params["df"]
            if "theta" in params:
                row["theta"] = params["theta"]
            rows.append(row)

        return pd.DataFrame(rows)

    def simulate_joint(
        self,
        family: str,
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Generate joint uniform samples from a specified fitted copula.

        Parameters
        ----------
        family : str
            Copula family name (must have been fitted already).
        n_samples : int | None
            Number of samples; defaults to ``config.simulation.n_samples``.

        Returns
        -------
        np.ndarray
            Uniform samples of shape (n_samples, d).
        """
        if family not in self._fitted_params:
            raise ValueError(
                f"Family '{family}' has not been fitted. "
                f"Available: {list(self._fitted_params.keys())}"
            )
        if n_samples is None:
            n_samples = self.n_samples

        return copula_sample(
            self._fitted_params[family], n_samples, seed=self.seed
        )

    def portfolio_var_by_copula(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        alphas: list[float],
    ) -> pd.DataFrame:
        """Compute portfolio VaR under each copula family.

        Pipeline for each family:
            1. Fit GARCH marginals and apply PIT.
            2. Fit the copula to pseudo-uniform data.
            3. Simulate 10,000 joint uniform samples.
            4. Invert PIT to obtain return scenarios.
            5. Compute portfolio returns and VaR at each alpha.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset log-returns (T x d).
        weights : np.ndarray
            Portfolio weights of length d.
        alphas : list[float]
            Confidence levels (e.g. [0.95, 0.99]).

        Returns
        -------
        pd.DataFrame
            Rows = copula families, columns = ``VaR_alpha`` for each alpha.
        """
        weights = np.asarray(weights, dtype=np.float64)

        # Step 1: filter marginals
        garch_dist = self.config["marginals"]["garch_dist"]
        u_data, garch_models = filter_marginals(returns, garch_dist=garch_dist)

        # Step 2: fit all copulas
        self.fit_all_families(u_data)

        # Forecast 1-step-ahead sigmas for inverse PIT
        forecast_sigmas = np.array(
            [_last_sigma(m) for m in garch_models], dtype=np.float64
        )

        rows = []
        for family in self.families:
            # Step 3: simulate from copula
            u_sim = self.simulate_joint(family, self.n_samples)

            # For Archimedean (bivariate), pad to full dimension with
            # independent uniforms for remaining assets
            d = returns.shape[1]
            if u_sim.shape[1] < d:
                rng = np.random.default_rng(self.seed + 1)
                extra = rng.uniform(size=(u_sim.shape[0], d - u_sim.shape[1]))
                u_sim = np.column_stack([u_sim, extra])

            # Step 4: invert PIT to get return scenarios
            return_scenarios = inverse_pit(u_sim, garch_models, forecast_sigmas)

            # Step 5: portfolio returns and VaR
            portfolio_returns = return_scenarios @ weights
            portfolio_losses = -portfolio_returns

            row = {"family": family}
            for alpha in alphas:
                var_val = float(np.quantile(portfolio_losses, alpha))
                row[f"VaR_{alpha}"] = var_val
            rows.append(row)

        return pd.DataFrame(rows)


def _last_sigma(fitted_model) -> float:
    """Extract the last 1-step-ahead conditional sigma (decimal form).

    The ``arch`` library stores conditional volatility in percentage
    units, so we divide by 100.
    """
    fcast = fitted_model.forecast(horizon=1)
    last_var_pct = float(fcast.variance.dropna().iloc[-1, 0])
    return np.sqrt(last_var_pct) / 100.0
