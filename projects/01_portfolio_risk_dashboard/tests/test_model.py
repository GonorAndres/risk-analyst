"""Unit tests for the Project 01 RiskModel.

Tests cover:
    - RiskModel with synthetic data (known normal distribution)
    - VaR/ES values against analytic expectations
    - Rolling VaR shape and NaN handling
    - Backtest on synthetic data with known violation rate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Add the project src to the path so we can import model.py
_PROJECT_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _PROJECT_SRC not in sys.path:
    sys.path.insert(0, _PROJECT_SRC)

from model import RiskModel

from risk_analyst.measures.backtesting import backtest_var

SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """Create synthetic normally-distributed returns for 3 assets over 1000 days.

    Means and volatilities are chosen to give predictable portfolio behavior.
    """
    rng = np.random.default_rng(SEED)
    n_days = 1000
    n_assets = 3
    # Daily means and vols
    mu = np.array([0.0003, 0.0002, 0.0001])
    sigma = np.array([0.015, 0.010, 0.008])

    data = rng.normal(
        loc=mu,
        scale=sigma,
        size=(n_days, n_assets),
    )
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


@pytest.fixture
def equal_weights() -> np.ndarray:
    return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])


@pytest.fixture
def fitted_model(
    synthetic_returns: pd.DataFrame,
    equal_weights: np.ndarray,
) -> RiskModel:
    """Fit a RiskModel on the synthetic data."""
    model = RiskModel(n_sims=10_000, seed=SEED)
    model.fit(synthetic_returns, equal_weights)
    return model


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestRiskModelBasic:
    """Basic sanity checks on the fitted model."""

    def test_losses_shape(
        self,
        fitted_model: RiskModel,
        synthetic_returns: pd.DataFrame,
    ) -> None:
        assert len(fitted_model.losses) == len(synthetic_returns)

    def test_losses_are_finite(self, fitted_model: RiskModel) -> None:
        assert np.all(np.isfinite(fitted_model.losses))

    def test_unfitted_raises(self) -> None:
        model = RiskModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.losses


# ---------------------------------------------------------------------------
# VaR/ES values
# ---------------------------------------------------------------------------


class TestRiskModelVaR:
    """VaR and ES on synthetic normal data."""

    @pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
    def test_var_is_finite(self, fitted_model: RiskModel, method: str) -> None:
        var_val = fitted_model.var(alpha=0.99, method=method)
        assert np.isfinite(var_val)

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
    def test_var_monotonicity(self, fitted_model: RiskModel, alpha: float) -> None:
        """Implicitly tested via the shared tests, but verify at model level."""
        if alpha > 0.90:
            var_lower = fitted_model.var(alpha=alpha - 0.05, method="historical")
            var_upper = fitted_model.var(alpha=alpha, method="historical")
            assert var_upper >= var_lower

    def test_es_geq_var(self, fitted_model: RiskModel) -> None:
        for alpha in [0.90, 0.95, 0.99]:
            var_val = fitted_model.var(alpha=alpha, method="historical")
            es_val = fitted_model.es(alpha=alpha)
            assert es_val >= var_val

    def test_parametric_var_close_to_normal(
        self,
        synthetic_returns: pd.DataFrame,
        equal_weights: np.ndarray,
    ) -> None:
        """For normal data, parametric VaR should be close to the true quantile."""
        # Generate a large sample for accuracy
        rng = np.random.default_rng(123)
        n = 50_000
        mu = np.array([0.0003, 0.0002, 0.0001])
        sigma = np.array([0.015, 0.010, 0.008])
        data = rng.normal(loc=mu, scale=sigma, size=(n, 3))
        dates = pd.bdate_range(start="2010-01-01", periods=n)
        returns = pd.DataFrame(data, index=dates, columns=["A", "B", "C"])
        w = equal_weights

        model = RiskModel(seed=SEED)
        model.fit(returns, w)
        var_param = model.var(alpha=0.99, method="parametric")

        # The portfolio return is sum(w_i * r_i).  For independent normals:
        #   mu_p = w' mu, sigma_p = sqrt(w' diag(sigma^2) w)
        mu_p = -(w @ mu)  # losses = -returns
        sigma_p = np.sqrt(w @ np.diag(sigma**2) @ w)
        analytic = mu_p + sigma_p * stats.norm.ppf(0.99)

        assert var_param == pytest.approx(analytic, abs=0.001)


# ---------------------------------------------------------------------------
# Rolling VaR
# ---------------------------------------------------------------------------


class TestRollingVaR:
    """Rolling VaR computation checks."""

    def test_rolling_var_shape(self, fitted_model: RiskModel) -> None:
        rv = fitted_model.rolling_var(window=100, alpha=0.95, method="historical")
        assert len(rv) == len(fitted_model.losses)

    def test_rolling_var_initial_nan(self, fitted_model: RiskModel) -> None:
        window = 100
        rv = fitted_model.rolling_var(window=window, alpha=0.95, method="historical")
        # First `window` entries should be NaN
        assert rv.iloc[:window].isna().all()
        # Entries from `window` onward should not be NaN
        assert rv.iloc[window:].notna().all()


# ---------------------------------------------------------------------------
# Backtesting on synthetic data
# ---------------------------------------------------------------------------


class TestBacktestSynthetic:
    """Backtest rolling VaR on synthetic data and verify violation rate."""

    def test_violation_rate_near_expected(self, fitted_model: RiskModel) -> None:
        """The observed violation rate should be in a reasonable range."""
        alpha = 0.95
        window = 252
        rv = fitted_model.rolling_var(window=window, alpha=alpha, method="historical")

        valid = ~rv.isna()
        losses_valid = fitted_model.losses[valid.values]
        var_valid = rv[valid].values

        report = backtest_var(
            losses=losses_valid,
            var_series=var_valid,
            alpha=alpha,
        )

        expected = 1.0 - alpha  # 0.05
        # Allow the violation rate to be within [0.01, 0.15] for 1000-day sample
        assert 0.01 <= report.violation_rate <= 0.15, (
            f"Violation rate {report.violation_rate:.3f} outside expected range "
            f"for alpha={alpha}"
        )

    def test_backtest_report_structure(self, fitted_model: RiskModel) -> None:
        alpha = 0.99
        window = 252
        rv = fitted_model.rolling_var(window=window, alpha=alpha, method="historical")

        valid = ~rv.isna()
        losses_valid = fitted_model.losses[valid.values]
        var_valid = rv[valid].values

        report = backtest_var(losses=losses_valid, var_series=var_valid, alpha=alpha)

        assert report.n_obs == len(losses_valid)
        assert report.n_violations >= 0
        assert 0.0 <= report.violation_rate <= 1.0
        assert report.kupiec.statistic >= 0.0
        assert 0.0 <= report.kupiec.p_value <= 1.0
        assert report.traffic_light.zone in ("green", "yellow", "red")
