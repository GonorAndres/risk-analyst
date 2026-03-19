"""Tests for Project 04 -- Volatility Modeling.

Covers GARCH fitting on synthetic data, conditional risk measures,
model comparison consistency, and Ljung-Box diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest
from diagnostics import ljung_box_test

# Project-local imports (conftest.py adds project src/ to sys.path)
from model import VolatilityModel

from risk_analyst.models.volatility import (
    conditional_es,
    conditional_var,
    fit_garch,
    fit_gjr_garch,
    forecast_volatility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_garch_data(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    seed: int,
) -> np.ndarray:
    """Generate synthetic GARCH(1,1) return series.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    r_t = sigma_t * z_t,   z_t ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return returns


# True GARCH(1,1) parameters (in decimal returns)
TRUE_OMEGA = 1e-6
TRUE_ALPHA = 0.08
TRUE_BETA = 0.90
SEED = 42
N = 5000

SYNTH_RETURNS = generate_garch_data(N, TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA, SEED)


# ---------------------------------------------------------------------------
# Tests: GARCH fitting on synthetic data
# ---------------------------------------------------------------------------

class TestGARCHFit:
    """Verify GARCH(1,1) parameter recovery on synthetic data."""

    @pytest.fixture(autouse=True)
    def _fit(self) -> None:
        # Use normal dist to match the DGP (Gaussian innovations)
        self.result = fit_garch(SYNTH_RETURNS, p=1, q=1, dist="normal")

    def test_alpha_recovery(self) -> None:
        """Recovered alpha[1] within 0.1 of true value."""
        alpha_hat = self.result.params["alpha[1]"]
        assert alpha_hat == pytest.approx(TRUE_ALPHA, abs=0.1), (
            f"alpha[1]={alpha_hat:.4f}, true={TRUE_ALPHA}"
        )

    def test_beta_recovery(self) -> None:
        """Recovered beta[1] within 0.1 of true value."""
        beta_hat = self.result.params["beta[1]"]
        assert beta_hat == pytest.approx(TRUE_BETA, abs=0.1), (
            f"beta[1]={beta_hat:.4f}, true={TRUE_BETA}"
        )


class TestConditionalVolatility:
    """Conditional volatility must always be positive."""

    def test_positive_volatility(self) -> None:
        result = fit_garch(SYNTH_RETURNS, p=1, q=1, dist="normal")
        cond_vol = result.conditional_volatility
        assert (cond_vol > 0).all(), "Conditional volatility has non-positive values."


class TestGJRGARCH:
    """GJR-GARCH should detect leverage (gamma >= 0 on equity-like data)."""

    def test_gamma_nonnegative(self) -> None:
        """On equity returns, the asymmetric coefficient should be >= 0."""
        # Create data with leverage: negative returns inflate vol more
        rng = np.random.default_rng(123)
        n = 3000
        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        omega, alpha, gamma, beta = 1e-6, 0.04, 0.08, 0.88
        sigma2[0] = omega / (1 - alpha - gamma / 2 - beta)
        for t in range(1, n):
            indicator = 1.0 if returns[t - 1] < 0 else 0.0
            sigma2[t] = (
                omega
                + alpha * returns[t - 1] ** 2
                + gamma * returns[t - 1] ** 2 * indicator
                + beta * sigma2[t - 1]
            )
            returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

        result = fit_gjr_garch(returns, p=1, o=1, q=1, dist="normal")
        gamma_hat = result.params["gamma[1]"]
        assert gamma_hat >= 0.0, f"gamma[1]={gamma_hat:.4f} should be >= 0."


class TestModelComparison:
    """AIC ordering is consistent: lower = better fit."""

    def test_aic_ordering(self) -> None:
        config = {
            "models": {
                "garch": {"p": 1, "q": 1, "dist": "normal"},
                "gjr_garch": {"p": 1, "o": 1, "q": 1, "dist": "normal"},
                "egarch": {"p": 1, "o": 1, "q": 1, "dist": "normal"},
            },
            "risk": {"confidence_levels": [0.95, 0.99], "forecast_horizon": 1},
            "random_seed": 42,
        }
        vm = VolatilityModel(config)
        comparison = vm.compare_models(SYNTH_RETURNS)

        # DataFrame is sorted by AIC ascending
        aic_values = comparison["AIC"].values
        assert all(
            aic_values[i] <= aic_values[i + 1] for i in range(len(aic_values) - 1)
        ), "compare_models() should return AIC-sorted rows."


class TestConditionalRisk:
    """Conditional VaR and ES properties."""

    @pytest.fixture(autouse=True)
    def _fit(self) -> None:
        self.result = fit_garch(SYNTH_RETURNS, p=1, q=1, dist="t")

    def test_var_positive(self) -> None:
        """Conditional VaR > 0 for high confidence level."""
        var = conditional_var(self.result, alpha=0.99)
        assert var > 0, f"VaR should be positive, got {var}"

    def test_es_geq_var(self) -> None:
        """ES >= VaR by definition (coherent risk measure)."""
        var = conditional_var(self.result, alpha=0.99)
        es = conditional_es(self.result, alpha=0.99)
        assert es >= var - 1e-10, f"ES={es:.6f} < VaR={var:.6f}"


class TestLjungBox:
    """Ljung-Box on squared standardized residuals."""

    def test_no_remaining_clustering(self) -> None:
        """After fitting GARCH to GARCH-generated data, squared std resids
        should show no significant autocorrelation (p > 0.05)."""
        result = fit_garch(SYNTH_RETURNS, p=1, q=1, dist="normal")
        raw = result.std_resid
        if hasattr(raw, "dropna"):
            std_resid = raw.dropna().values
        else:
            arr = np.asarray(raw, dtype=np.float64)
            std_resid = arr[~np.isnan(arr)]
        lb = ljung_box_test(std_resid, lags=10)
        # Check that at least the first few lags are not significant
        min_pvalue = lb["lb_pvalue"].min()
        assert min_pvalue > 0.01, (
            f"Ljung-Box min p-value={min_pvalue:.4f}; GARCH should remove clustering."
        )


class TestForecast:
    """Volatility forecasts should be positive and well-shaped."""

    def test_forecast_positive(self) -> None:
        result = fit_garch(SYNTH_RETURNS, p=1, q=1, dist="normal")
        fc = forecast_volatility(result, horizon=5)
        assert (fc.values > 0).all(), "All forecast values must be positive."

    def test_forecast_shape(self) -> None:
        result = fit_garch(SYNTH_RETURNS, p=1, q=1, dist="normal")
        fc = forecast_volatility(result, horizon=10)
        assert fc.shape[1] == 10, "Horizon=10 should produce 10 columns."


class TestVolatilityModelOrchestrator:
    """Integration tests for the VolatilityModel class."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        config = {
            "models": {
                "garch": {"p": 1, "q": 1, "dist": "normal"},
                "gjr_garch": {"p": 1, "o": 1, "q": 1, "dist": "normal"},
                "egarch": {"p": 1, "o": 1, "q": 1, "dist": "normal"},
            },
            "risk": {"confidence_levels": [0.95, 0.99], "forecast_horizon": 5},
            "random_seed": 42,
        }
        self.vm = VolatilityModel(config)
        self.vm.fit_all(SYNTH_RETURNS)

    def test_fit_all_keys(self) -> None:
        assert set(self.vm._fitted_models.keys()) == {
            "garch", "gjr_garch", "egarch"
        }

    def test_compute_conditional_risk(self) -> None:
        risk = self.vm.compute_conditional_risk("garch", alpha=0.99)
        assert "VaR" in risk and "ES" in risk
        assert risk["VaR"] > 0
        assert risk["ES"] >= risk["VaR"] - 1e-10

    def test_fhs(self) -> None:
        fhs = self.vm.filtered_historical_simulation(
            SYNTH_RETURNS, "garch", alpha=0.99, n_sims=5000
        )
        assert fhs["VaR"] > 0
        assert fhs["ES"] >= fhs["VaR"] - 1e-10

    def test_forecast_from_config(self) -> None:
        fc = self.vm.forecast("garch")
        assert fc.shape[1] == 5  # from config forecast_horizon
