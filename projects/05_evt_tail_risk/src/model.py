"""EVT Model: block maxima (GEV) and peaks-over-threshold (GPD) analysis.

Provides the ``EVTModel`` class that orchestrates threshold selection,
GPD/GEV fitting, EVT-based risk measures, and method comparison.

References:
    - de Haan & Ferreira (2006), Ch. 1--3: GEV and GPD theory.
    - McNeil & Frey (2000): tail-related risk measures for financial series.
    - Coles (2001), Ch. 3--4: block maxima and POT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Project-local import (threshold.py lives in the same src/ directory)
from threshold import select_threshold_auto

from risk_analyst.models.evt import (
    evt_es,
    evt_var,
    fit_gev,
    fit_gpd,
    return_level,
)


class EVTModel:
    """Extreme Value Theory model for tail risk analysis.

    Supports two approaches:
        1. Block maxima -> GEV fitting -> return levels
        2. Peaks-over-threshold -> GPD fitting -> VaR / ES

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically loaded from default.yaml).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._gev_params: dict | None = None
        self._gpd_params: dict | None = None
        self._n_total: int | None = None
        self._block_maxima: np.ndarray | None = None

    def fit_block_maxima(
        self, losses: np.ndarray, block_size: int
    ) -> dict:
        """Split losses into blocks, extract block maxima, and fit GEV.

        Parameters
        ----------
        losses : np.ndarray
            1-D array of loss observations.
        block_size : int
            Number of observations per block (e.g. 21 for monthly).

        Returns
        -------
        dict
            GEV fit results from :func:`fit_gev`, plus ``n_blocks`` key.
        """
        n = len(losses)
        n_blocks = n // block_size

        # Trim to exact multiple of block_size
        trimmed = losses[: n_blocks * block_size]
        blocks = trimmed.reshape(n_blocks, block_size)
        block_max = blocks.max(axis=1)

        self._block_maxima = block_max
        self._gev_params = fit_gev(block_max)
        self._gev_params["n_blocks"] = n_blocks

        return self._gev_params

    def fit_pot(
        self, losses: np.ndarray, threshold: float | None = None
    ) -> dict:
        """Peaks-over-threshold: extract exceedances and fit GPD.

        Parameters
        ----------
        losses : np.ndarray
            1-D array of loss observations.
        threshold : float | None
            Threshold value u. If None, automatically selected using
            the method specified in config.

        Returns
        -------
        dict
            GPD fit results from :func:`fit_gpd`.
        """
        self._n_total = len(losses)

        if threshold is None:
            pot_cfg = self.config.get("pot", {})
            method = pot_cfg.get("threshold_method", "percentile")
            pct = pot_cfg.get("threshold_percentile", 95.0)
            threshold = select_threshold_auto(
                losses, method=method, percentile=pct
            )

        exceedances = losses[losses > threshold] - threshold
        self._gpd_params = fit_gpd(exceedances, threshold)

        return self._gpd_params

    def compute_risk(self, alpha: float) -> dict:
        """Compute EVT-based VaR and ES at a given confidence level.

        Parameters
        ----------
        alpha : float
            Confidence level (e.g. 0.99).

        Returns
        -------
        dict
            Keys: alpha, var_evt, es_evt.

        Raises
        ------
        RuntimeError
            If :meth:`fit_pot` has not been called first.
        """
        if self._gpd_params is None or self._n_total is None:
            raise RuntimeError(
                "Must call fit_pot() before compute_risk()."
            )

        var = evt_var(self._gpd_params, self._n_total, alpha)
        es = evt_es(self._gpd_params, self._n_total, alpha)

        return {"alpha": alpha, "var_evt": var, "es_evt": es}

    def compare_methods(
        self, losses: np.ndarray, alphas: list[float]
    ) -> pd.DataFrame:
        """Compare EVT VaR/ES with normal and t-distribution estimates.

        Parameters
        ----------
        losses : np.ndarray
            1-D array of loss observations.
        alphas : list[float]
            List of confidence levels to compare.

        Returns
        -------
        pd.DataFrame
            Columns: alpha, var_normal, var_t, var_evt,
                     es_normal, es_t, es_evt.

        Raises
        ------
        RuntimeError
            If :meth:`fit_pot` has not been called first.
        """
        if self._gpd_params is None or self._n_total is None:
            raise RuntimeError(
                "Must call fit_pot() before compare_methods()."
            )

        mu_l = float(np.mean(losses))
        sigma_l = float(np.std(losses, ddof=1))

        # Fit Student-t to losses
        df_t, loc_t, scale_t = stats.t.fit(losses)

        records = []
        for alpha in alphas:
            # Normal VaR and ES
            z_alpha = stats.norm.ppf(alpha)
            var_normal = mu_l + sigma_l * z_alpha
            # Normal ES: mu + sigma * phi(z) / (1 - alpha)
            es_normal = mu_l + sigma_l * (
                stats.norm.pdf(z_alpha) / (1 - alpha)
            )

            # Student-t VaR and ES
            t_alpha = stats.t.ppf(alpha, df_t)
            var_t = loc_t + scale_t * t_alpha
            # Student-t ES: loc + scale * (t_pdf(t_alpha) * (df+t_alpha^2) / ((df-1)*(1-alpha)))
            t_pdf_val = stats.t.pdf(t_alpha, df_t)
            es_t = loc_t + scale_t * (
                t_pdf_val * (df_t + t_alpha**2) / ((df_t - 1) * (1 - alpha))
            )

            # EVT VaR and ES
            var_evt = evt_var(self._gpd_params, self._n_total, alpha)
            es_evt = evt_es(self._gpd_params, self._n_total, alpha)

            records.append({
                "alpha": alpha,
                "var_normal": var_normal,
                "var_t": var_t,
                "var_evt": var_evt,
                "es_normal": es_normal,
                "es_t": es_t,
                "es_evt": es_evt,
            })

        return pd.DataFrame(records)

    def return_levels(
        self, return_periods: list[float]
    ) -> pd.DataFrame:
        """Compute return levels for given return periods.

        Parameters
        ----------
        return_periods : list[float]
            List of return periods T (in units of blocks).

        Returns
        -------
        pd.DataFrame
            Columns: return_period, return_level.

        Raises
        ------
        RuntimeError
            If :meth:`fit_block_maxima` has not been called first.
        """
        if self._gev_params is None:
            raise RuntimeError(
                "Must call fit_block_maxima() before return_levels()."
            )

        records = []
        for T in return_periods:
            rl = return_level(self._gev_params, T)
            records.append({
                "return_period": T,
                "return_level": rl,
            })

        return pd.DataFrame(records)
