"""Project 04 -- Volatility Modeling: core model orchestration.

Provides the ``VolatilityModel`` class that fits, compares, and
forecasts across multiple GARCH-family specifications, and implements
Filtered Historical Simulation (FHS) for VaR estimation.

References:
    - Barone-Adesi, Giannopoulos & Vosper (1999): FHS.
    - Hansen & Lunde (2005): comparison of volatility models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from arch.univariate.base import ARCHModelResult

from risk_analyst.models.volatility import (
    conditional_es,
    conditional_var,
    fit_egarch,
    fit_garch,
    fit_gjr_garch,
    forecast_volatility,
)


class VolatilityModel:
    """Orchestrates fitting, comparison, and forecasting across GARCH variants.

    Parameters
    ----------
    config : dict | str | Path
        Either a dict with the configuration structure, or a path to a
        YAML file matching ``configs/default.yaml``.
    """

    def __init__(self, config: dict[str, Any] | str | Path) -> None:
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config: dict[str, Any] = yaml.safe_load(f)
        else:
            self.config = config

        self._fitted_models: dict[str, ARCHModelResult] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_all(
        self,
        returns: np.ndarray | pd.Series,
    ) -> dict[str, ARCHModelResult]:
        """Fit GARCH, GJR-GARCH, and EGARCH models from config.

        Parameters
        ----------
        returns : array-like
            Decimal return series.

        Returns
        -------
        dict[str, ARCHModelResult]
            Mapping ``{"garch": ..., "gjr_garch": ..., "egarch": ...}``.
        """
        models_cfg = self.config["models"]

        garch_cfg = models_cfg["garch"]
        self._fitted_models["garch"] = fit_garch(
            returns,
            p=garch_cfg["p"],
            q=garch_cfg["q"],
            dist=garch_cfg["dist"],
        )

        gjr_cfg = models_cfg["gjr_garch"]
        self._fitted_models["gjr_garch"] = fit_gjr_garch(
            returns,
            p=gjr_cfg["p"],
            o=gjr_cfg["o"],
            q=gjr_cfg["q"],
            dist=gjr_cfg["dist"],
        )

        egarch_cfg = models_cfg["egarch"]
        self._fitted_models["egarch"] = fit_egarch(
            returns,
            p=egarch_cfg["p"],
            o=egarch_cfg["o"],
            q=egarch_cfg["q"],
            dist=egarch_cfg["dist"],
        )

        return dict(self._fitted_models)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        returns: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Compare fitted models by AIC, BIC, and log-likelihood.

        If models have not yet been fitted, ``fit_all`` is called first.

        Parameters
        ----------
        returns : array-like
            Decimal return series (used if models are not yet fitted).

        Returns
        -------
        pd.DataFrame
            Comparison table with columns ``["AIC", "BIC", "Log-Likelihood"]``,
            indexed by model name, sorted by AIC ascending (best first).
        """
        if not self._fitted_models:
            self.fit_all(returns)

        records: list[dict[str, Any]] = []
        for name, model in self._fitted_models.items():
            records.append(
                {
                    "model": name,
                    "AIC": model.aic,
                    "BIC": model.bic,
                    "Log-Likelihood": model.loglikelihood,
                }
            )

        df = pd.DataFrame(records).set_index("model").sort_values("AIC")
        return df

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        model_name: str,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        """Produce volatility forecast from a fitted model.

        Parameters
        ----------
        model_name : str
            One of ``"garch"``, ``"gjr_garch"``, ``"egarch"``.
        horizon : int | None
            Forecast horizon.  Falls back to ``config["risk"]["forecast_horizon"]``.

        Returns
        -------
        pd.DataFrame
            Forecasted conditional standard deviation (decimal) at each step.
        """
        if model_name not in self._fitted_models:
            raise KeyError(
                f"Model '{model_name}' not fitted. Call fit_all() first."
            )
        if horizon is None:
            horizon = self.config["risk"]["forecast_horizon"]

        return forecast_volatility(self._fitted_models[model_name], horizon=horizon)

    # ------------------------------------------------------------------
    # Conditional risk measures
    # ------------------------------------------------------------------

    def compute_conditional_risk(
        self,
        model_name: str,
        alpha: float,
    ) -> dict[str, float]:
        """Compute conditional VaR and ES for a fitted model.

        Parameters
        ----------
        model_name : str
            Fitted model key.
        alpha : float
            Confidence level (e.g. 0.99).

        Returns
        -------
        dict
            ``{"VaR": ..., "ES": ...}`` in decimal returns.
        """
        if model_name not in self._fitted_models:
            raise KeyError(
                f"Model '{model_name}' not fitted. Call fit_all() first."
            )
        fitted = self._fitted_models[model_name]
        var = conditional_var(fitted, alpha)
        es = conditional_es(fitted, alpha)
        return {"VaR": var, "ES": es}

    # ------------------------------------------------------------------
    # Filtered Historical Simulation (FHS)
    # ------------------------------------------------------------------

    def filtered_historical_simulation(
        self,
        returns: np.ndarray | pd.Series,
        model_name: str,
        alpha: float,
        n_sims: int = 10_000,
    ) -> dict[str, float]:
        """Filtered Historical Simulation for VaR and ES.

        FHS procedure (Barone-Adesi et al., 1999):
        1. Fit GARCH to the return series -> obtain sigma_t for each t.
        2. Compute standardized residuals: z_t = r_t / sigma_t.
        3. Resample z_t with replacement -> z_t^*.
        4. Rescale by the current (last) conditional sigma:
           r_t^* = sigma_T * z_t^*.
        5. Compute VaR and ES from the simulated return distribution.

        Parameters
        ----------
        returns : array-like
            Decimal return series.
        model_name : str
            Which fitted model to use for sigma_t.
        alpha : float
            Confidence level.
        n_sims : int
            Number of bootstrap draws.

        Returns
        -------
        dict
            ``{"VaR": ..., "ES": ...}`` from the simulated distribution.
        """
        if model_name not in self._fitted_models:
            raise KeyError(
                f"Model '{model_name}' not fitted. Call fit_all() first."
            )

        fitted = self._fitted_models[model_name]
        seed = self.config.get("random_seed", 42)
        rng = np.random.default_rng(seed)

        # Standardized residuals from the fitted model
        raw_resid = fitted.std_resid
        if isinstance(raw_resid, pd.Series):
            std_resid = raw_resid.dropna().values
        else:
            arr = np.asarray(raw_resid, dtype=np.float64)
            std_resid = arr[~np.isnan(arr)]

        # Current conditional sigma (latest 1-step-ahead, decimal)
        fcast = fitted.forecast(horizon=1)
        sigma_T = np.sqrt(float(fcast.variance.dropna().iloc[-1, 0])) / 100.0

        # Bootstrap: resample standardized residuals, rescale
        z_star = rng.choice(std_resid, size=n_sims, replace=True)
        simulated_returns = sigma_T * z_star

        # Losses = -returns (positive = loss)
        simulated_losses = -simulated_returns

        var = float(np.quantile(simulated_losses, alpha))
        tail = simulated_losses[simulated_losses >= var]
        es = float(np.mean(tail)) if len(tail) > 0 else var

        return {"VaR": var, "ES": es}
