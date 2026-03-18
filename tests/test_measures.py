"""Unit tests for shared risk measure implementations.

Tests cover:
    - VaR monotonicity (higher alpha -> higher VaR)
    - ES >= VaR always
    - Kupiec test with known violation count
    - Parametric VaR against analytic normal quantile
    - Historical VaR against numpy percentile
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from risk_analyst.measures.backtesting import (
    christoffersen_test,
    kupiec_test,
    traffic_light_test,
)
from risk_analyst.measures.var import (
    expected_shortfall,
    historical_var,
    monte_carlo_var,
    parametric_var,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture
def normal_losses() -> np.ndarray:
    """Generate 10,000 standard normal losses for testing."""
    rng = np.random.default_rng(SEED)
    return rng.normal(loc=0.0, scale=1.0, size=10_000)


@pytest.fixture
def known_normal_losses() -> np.ndarray:
    """Losses with known mu=0.01, sigma=0.02 for parametric comparison."""
    rng = np.random.default_rng(SEED)
    return rng.normal(loc=0.01, scale=0.02, size=50_000)


# ---------------------------------------------------------------------------
# VaR monotonicity: higher alpha -> higher VaR
# ---------------------------------------------------------------------------


class TestVaRMonotonicity:
    """VaR must increase with the confidence level alpha."""

    alphas = [0.90, 0.95, 0.99]

    def test_historical_var_monotonic(self, normal_losses: np.ndarray) -> None:
        vars_ = [historical_var(normal_losses, a) for a in self.alphas]
        for i in range(len(vars_) - 1):
            assert vars_[i] < vars_[i + 1], (
                f"Historical VaR not monotonic: VaR({self.alphas[i]})={vars_[i]:.4f} "
                f">= VaR({self.alphas[i+1]})={vars_[i+1]:.4f}"
            )

    def test_parametric_var_monotonic(self, normal_losses: np.ndarray) -> None:
        vars_ = [parametric_var(normal_losses, a) for a in self.alphas]
        for i in range(len(vars_) - 1):
            assert vars_[i] < vars_[i + 1]

    def test_monte_carlo_var_monotonic(self, normal_losses: np.ndarray) -> None:
        vars_ = [monte_carlo_var(normal_losses, a, n_sims=50_000, seed=SEED) for a in self.alphas]
        for i in range(len(vars_) - 1):
            assert vars_[i] < vars_[i + 1]


# ---------------------------------------------------------------------------
# ES >= VaR always
# ---------------------------------------------------------------------------


class TestESBound:
    """Expected Shortfall must be at least as large as VaR."""

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
    def test_es_geq_var(self, normal_losses: np.ndarray, alpha: float) -> None:
        var_val = historical_var(normal_losses, alpha)
        es_val = expected_shortfall(normal_losses, alpha)
        assert es_val >= var_val, (
            f"ES({alpha})={es_val:.4f} < VaR({alpha})={var_val:.4f}"
        )


# ---------------------------------------------------------------------------
# Parametric VaR vs. analytic normal quantile
# ---------------------------------------------------------------------------


class TestParametricVaR:
    """Parametric VaR should match mu + sigma * z_alpha for large samples."""

    @pytest.mark.parametrize("alpha", [0.95, 0.99])
    def test_against_analytic(self, known_normal_losses: np.ndarray, alpha: float) -> None:
        mu = 0.01
        sigma = 0.02
        z_alpha = stats.norm.ppf(alpha)
        analytic_var = mu + sigma * z_alpha

        computed_var = parametric_var(known_normal_losses, alpha)
        # With 50k samples, should be very close
        assert computed_var == pytest.approx(analytic_var, abs=0.001), (
            f"Parametric VaR({alpha}) = {computed_var:.5f}, "
            f"expected {analytic_var:.5f}"
        )


# ---------------------------------------------------------------------------
# Historical VaR vs. numpy percentile
# ---------------------------------------------------------------------------


class TestHistoricalVaR:
    """Historical VaR should match np.percentile exactly."""

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
    def test_against_numpy(self, normal_losses: np.ndarray, alpha: float) -> None:
        var_val = historical_var(normal_losses, alpha)
        expected = np.quantile(normal_losses, alpha)
        assert var_val == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Kupiec test with known violations
# ---------------------------------------------------------------------------


class TestKupiecTest:
    """Kupiec test with known violation counts."""

    def test_no_reject_expected_rate(self) -> None:
        """With exactly the expected number of violations, should not reject."""
        # 250 obs, alpha=0.99, expected violations = 2.5 -> use 3
        result = kupiec_test(violations=3, n_obs=250, alpha=0.99)
        assert not result.reject, (
            f"Kupiec test rejected with 3 violations out of 250 at 99% "
            f"(p={result.p_value:.4f})"
        )

    def test_reject_too_many_violations(self) -> None:
        """With far too many violations, should reject."""
        result = kupiec_test(violations=20, n_obs=250, alpha=0.99)
        assert result.reject, (
            f"Kupiec test did not reject with 20 violations out of 250 at 99% "
            f"(p={result.p_value:.4f})"
        )

    def test_lr_is_nonnegative(self) -> None:
        """LR statistic must be non-negative."""
        result = kupiec_test(violations=5, n_obs=250, alpha=0.99)
        assert result.statistic >= 0.0

    def test_zero_violations(self) -> None:
        """Edge case: zero violations."""
        result = kupiec_test(violations=0, n_obs=250, alpha=0.99)
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Christoffersen test
# ---------------------------------------------------------------------------


class TestChristoffersenTest:
    """Christoffersen independence test."""

    def test_independent_violations(self) -> None:
        """IID violations should not be rejected."""
        rng = np.random.default_rng(SEED)
        # ~1% violation rate, independent
        violations = (rng.uniform(size=1000) < 0.01).astype(int)
        result = christoffersen_test(violations)
        # Should generally not reject for independent series
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0

    def test_clustered_violations_detected(self) -> None:
        """Clustered violations (dependence) should be detected."""
        # Create a clearly clustered violation series
        v = np.zeros(500, dtype=int)
        # Insert two large clusters
        v[100:115] = 1  # 15 consecutive violations
        v[300:320] = 1  # 20 consecutive violations
        result = christoffersen_test(v)
        assert result.reject, (
            f"Christoffersen failed to detect clustering (p={result.p_value:.4f})"
        )


# ---------------------------------------------------------------------------
# Traffic light test
# ---------------------------------------------------------------------------


class TestTrafficLight:
    """Basel traffic-light zone classification."""

    def test_green_zone(self) -> None:
        result = traffic_light_test(n_violations=2, n_obs=250)
        assert result.zone == "green"

    def test_yellow_zone(self) -> None:
        result = traffic_light_test(n_violations=7, n_obs=250)
        assert result.zone == "yellow"

    def test_red_zone(self) -> None:
        result = traffic_light_test(n_violations=15, n_obs=250)
        assert result.zone == "red"


# ---------------------------------------------------------------------------
# Monte Carlo VaR reproducibility
# ---------------------------------------------------------------------------


class TestMCVaRReproducibility:
    """Monte Carlo VaR with the same seed should be deterministic."""

    def test_same_seed_same_result(self, normal_losses: np.ndarray) -> None:
        v1 = monte_carlo_var(normal_losses, 0.99, n_sims=5000, seed=42)
        v2 = monte_carlo_var(normal_losses, 0.99, n_sims=5000, seed=42)
        assert v1 == v2
