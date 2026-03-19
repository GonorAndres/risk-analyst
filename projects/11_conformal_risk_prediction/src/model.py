"""Main conformal risk model orchestrating VaR, PD, and comparison experiments.

Ties together the shared conformal library, project-specific models, and
adaptive inference into a single high-level interface driven by YAML config.

References:
    - Angelopoulos et al. (2024), Conformal risk control, JMLR.
    - Romano et al. (2019), Conformalized quantile regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from risk_analyst.measures.var import historical_var, parametric_var

from .adaptive import generate_regime_data, run_aci_experiment
from .models import ConformalPD, ConformalVaR


class ConformalRiskModel:
    """High-level orchestrator for conformal risk prediction experiments.

    Parameters
    ----------
    config : dict or str or Path
        Configuration dictionary or path to YAML config file.
    """

    def __init__(self, config: dict | str | Path) -> None:
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        self._seed: int = self.config.get("random_seed", 42)
        self._rng = np.random.default_rng(self._seed)

    def fit_var_intervals(
        self, returns: np.ndarray, alpha: float | None = None
    ) -> dict[str, Any]:
        """Compute conformal VaR intervals with parametric comparison.

        Parameters
        ----------
        returns : np.ndarray
            Return series.
        alpha : float or None
            Miscoverage level. Defaults to config value.

        Returns
        -------
        dict
            Keys: 'conformal_lower', 'conformal_upper', 'parametric_var',
            'historical_var', 'conformal_coverage'.
        """
        if alpha is None:
            alpha = self.config["conformal"]["alpha"]

        n = len(returns)
        cal_frac = self.config["conformal"]["cal_fraction"]
        n_train = int(n * (1 - cal_frac) * 0.6)
        n_cal = int(n * cal_frac)
        n_test = n - n_train - n_cal

        train = returns[:n_train]
        cal = returns[n_train : n_train + n_cal]
        test = returns[n_train + n_cal :]

        # Conformal VaR
        cvar = ConformalVaR(
            alpha=alpha,
            n_estimators=self.config.get("cqr", {}).get("n_estimators", 100),
            max_depth=self.config.get("cqr", {}).get("max_depth", 4),
            seed=self._seed,
        )
        cvar.fit(train, cal)
        lower, upper = cvar.predict_interval(test)
        coverage = cvar.coverage(test)

        # Parametric comparison
        losses = -returns  # convention: positive = loss
        param_var = parametric_var(losses, 1 - alpha)
        hist_var = historical_var(losses, 1 - alpha)

        return {
            "conformal_lower": lower,
            "conformal_upper": upper,
            "parametric_var": param_var,
            "historical_var": hist_var,
            "conformal_coverage": coverage,
        }

    def fit_pd_intervals(
        self, X: np.ndarray, y: np.ndarray, alpha: float | None = None
    ) -> dict[str, Any]:
        """Compute conformal PD intervals (bridge to P03 credit scoring).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Default indicators (0/1).
        alpha : float or None
            Miscoverage level.

        Returns
        -------
        dict
            Keys: 'pd_lower', 'pd_upper', 'pd_point', 'coverage'.
        """
        if alpha is None:
            alpha = self.config["conformal"]["alpha"]

        n = len(X)
        cal_frac = self.config["conformal"]["cal_fraction"]
        n_train = int(n * (1 - cal_frac))
        n_cal = int(n * cal_frac * 0.5)

        X_train, y_train = X[:n_train], y[:n_train]
        X_cal, y_cal = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
        X_test, y_test = X[n_train + n_cal :], y[n_train + n_cal :]

        cpd = ConformalPD(alpha=alpha, seed=self._seed)
        cpd.fit(X_train, y_train, X_cal, y_cal)

        lower, upper = cpd.predict_interval(X_test)
        coverage = cpd.coverage(X_test, y_test)
        pd_point = cpd._classifier.predict_proba(X_test)[:, 1]

        return {
            "pd_lower": lower,
            "pd_upper": upper,
            "pd_point": pd_point,
            "coverage": coverage,
            "y_test": y_test,
        }

    def compare_methods(
        self, returns: np.ndarray, alpha: float | None = None
    ) -> pd.DataFrame:
        """Compare conformal vs parametric vs bootstrap interval methods.

        Parameters
        ----------
        returns : np.ndarray
            Return series.
        alpha : float or None
            Miscoverage level.

        Returns
        -------
        pd.DataFrame
            Columns: method, coverage, avg_width, median_width.
        """
        if alpha is None:
            alpha = self.config["conformal"]["alpha"]

        n = len(returns)
        cal_frac = self.config["conformal"]["cal_fraction"]
        n_train = int(n * (1 - cal_frac) * 0.6)
        n_cal = int(n * cal_frac)

        train = returns[:n_train]
        cal = returns[n_train : n_train + n_cal]
        test = returns[n_train + n_cal :]

        results = []

        # --- Conformal (CQR) ---
        cvar = ConformalVaR(
            alpha=alpha,
            n_estimators=self.config.get("cqr", {}).get("n_estimators", 100),
            max_depth=self.config.get("cqr", {}).get("max_depth", 4),
            seed=self._seed,
        )
        cvar.fit(train, cal)
        c_lower, c_upper = cvar.predict_interval(test)
        _, y_test = ConformalVaR._make_features(test)
        c_covered = (y_test >= c_lower) & (y_test <= c_upper)
        c_widths = c_upper - c_lower
        results.append({
            "method": "conformal",
            "coverage": float(np.mean(c_covered)),
            "avg_width": float(np.mean(c_widths)),
            "median_width": float(np.median(c_widths)),
        })

        # --- Parametric (normal) ---
        mu = np.mean(train)
        sigma = np.std(train, ddof=1)
        z = stats.norm.ppf(1 - alpha / 2)
        p_lower = np.full_like(y_test, mu - z * sigma)
        p_upper = np.full_like(y_test, mu + z * sigma)
        p_covered = (y_test >= p_lower) & (y_test <= p_upper)
        p_widths = p_upper - p_lower
        results.append({
            "method": "parametric",
            "coverage": float(np.mean(p_covered)),
            "avg_width": float(np.mean(p_widths)),
            "median_width": float(np.median(p_widths)),
        })

        # --- Bootstrap ---
        rng = np.random.default_rng(self._seed)
        n_boot = 1000
        boot_means = np.array([
            np.mean(rng.choice(train, size=len(train), replace=True))
            for _ in range(n_boot)
        ])
        b_lower_val = np.percentile(boot_means, 100 * alpha / 2)
        b_upper_val = np.percentile(boot_means, 100 * (1 - alpha / 2))
        # Bootstrap intervals for each observation: center on prediction, scale by boot spread
        boot_half_width = (b_upper_val - b_lower_val) / 2 + sigma
        b_lower = np.full_like(y_test, mu - boot_half_width)
        b_upper = np.full_like(y_test, mu + boot_half_width)
        b_covered = (y_test >= b_lower) & (y_test <= b_upper)
        b_widths = b_upper - b_lower
        results.append({
            "method": "bootstrap",
            "coverage": float(np.mean(b_covered)),
            "avg_width": float(np.mean(b_widths)),
            "median_width": float(np.median(b_widths)),
        })

        return pd.DataFrame(results)

    def run_adaptive(
        self,
        returns: np.ndarray | None = None,
        alpha: float | None = None,
        gamma: float | None = None,
    ) -> dict:
        """Run ACI experiment with regime change data.

        Parameters
        ----------
        returns : np.ndarray or None
            Data to use. If None, generates regime-change data from config.
        alpha : float or None
            Target miscoverage level.
        gamma : float or None
            ACI step size.

        Returns
        -------
        dict
            ACI experiment results including coverage trajectory.
        """
        if alpha is None:
            alpha = self.config["conformal"]["alpha"]
        if gamma is None:
            gamma = self.config["adaptive"]["gamma"]

        if returns is None:
            returns = generate_regime_data(
                n_calm=self.config["adaptive"]["n_calm"],
                n_crisis=self.config["adaptive"]["n_crisis"],
                seed=self._seed,
            )

        # Simple rolling mean model
        def model_fn(data: np.ndarray) -> np.ndarray:
            """One-step-ahead prediction using expanding mean."""
            preds = np.zeros(len(data))
            for i in range(1, len(data)):
                preds[i] = np.mean(data[:i])
            return preds

        cal_size = min(100, len(returns) // 4)
        result = run_aci_experiment(
            data=returns,
            model_fn=model_fn,
            alpha=alpha,
            gamma=gamma,
            cal_size=cal_size,
        )
        result["n_calm"] = self.config["adaptive"]["n_calm"]
        result["n_crisis"] = self.config["adaptive"]["n_crisis"]
        return result
