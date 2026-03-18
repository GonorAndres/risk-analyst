"""Unit tests for copula dependency modeling (Project 06).

Uses SYNTHETIC data only -- no yfinance or arch dependency.
Generates correlated normals, applies PIT, and tests all copula
families for correct fitting, sampling, and tail dependence.

References:
    - Nelsen (2006), Ch. 4--5: copula properties.
    - McNeil, Frey & Embrechts (2015), Ch. 7: tail dependence formulas.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

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


# ---------------------------------------------------------------------------
# Fixtures: synthetic correlated data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_uniform_data() -> np.ndarray:
    """Correlated bivariate uniform data via normal copula."""
    rng = np.random.default_rng(42)
    n = 2000
    rho = 0.6
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    u_data = np.column_stack([stats.norm.cdf(z1), stats.norm.cdf(z2)])
    return u_data


@pytest.fixture
def rho() -> float:
    """True correlation used to generate synthetic data."""
    return 0.6


@pytest.fixture
def multivariate_uniform_data() -> np.ndarray:
    """Correlated 5-dimensional uniform data via Gaussian copula."""
    rng = np.random.default_rng(42)
    n = 2000
    d = 5
    # Random correlation matrix via random lower-triangular
    A = rng.standard_normal((d, d))
    cov = A @ A.T
    D = np.sqrt(np.diag(cov))
    corr = cov / np.outer(D, D)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, d)) @ L.T
    return stats.norm.cdf(z)


# ---------------------------------------------------------------------------
# 1. PIT produces values in (0, 1)
# ---------------------------------------------------------------------------

class TestPIT:

    def test_pit_values_in_unit_interval(self) -> None:
        """PIT output must lie strictly in (0, 1)."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((500, 3))
        u = pit_transform(data, method="empirical")
        assert np.all(u > 0)
        assert np.all(u < 1)

    def test_pit_uniform_ks_test(self) -> None:
        """PIT of any continuous data should be approximately uniform.

        Kolmogorov-Smirnov test: p-value > 0.05 for each column.
        """
        rng = np.random.default_rng(123)
        data = rng.standard_normal((1000, 2))
        u = pit_transform(data, method="empirical")
        for j in range(u.shape[1]):
            _, p_value = stats.kstest(u[:, j], "uniform")
            assert p_value > 0.05, (
                f"Column {j}: KS p-value = {p_value:.4f} <= 0.05"
            )


# ---------------------------------------------------------------------------
# 3--7. Tail dependence by family
# ---------------------------------------------------------------------------

class TestTailDependence:

    def test_gaussian_tail_dependence_zero(
        self, synthetic_uniform_data: np.ndarray
    ) -> None:
        """Gaussian copula: lambda_L = lambda_U = 0 (asymptotic independence)."""
        params = gaussian_copula_fit(synthetic_uniform_data)
        td = tail_dependence(params)
        assert td["lambda_L"] == pytest.approx(0.0, abs=1e-12)
        assert td["lambda_U"] == pytest.approx(0.0, abs=1e-12)

    def test_t_copula_tail_dependence_positive(
        self, synthetic_uniform_data: np.ndarray
    ) -> None:
        """t-copula: tail dependence > 0 for finite df."""
        params = t_copula_fit(synthetic_uniform_data)
        td = tail_dependence(params)
        assert td["lambda_L"] > 0, "t-copula lambda_L should be positive"
        assert td["lambda_U"] > 0, "t-copula lambda_U should be positive"
        # Symmetry
        assert td["lambda_L"] == pytest.approx(td["lambda_U"], abs=1e-12)

    def test_clayton_tail_dependence(
        self, synthetic_uniform_data: np.ndarray
    ) -> None:
        """Clayton: lambda_L > 0, lambda_U = 0."""
        params = clayton_copula_fit(synthetic_uniform_data)
        td = tail_dependence(params)
        assert td["lambda_L"] > 0, "Clayton lambda_L should be positive"
        assert td["lambda_U"] == pytest.approx(0.0, abs=1e-12)

    def test_gumbel_tail_dependence(
        self, synthetic_uniform_data: np.ndarray
    ) -> None:
        """Gumbel: lambda_L = 0, lambda_U > 0."""
        params = gumbel_copula_fit(synthetic_uniform_data)
        td = tail_dependence(params)
        assert td["lambda_L"] == pytest.approx(0.0, abs=1e-12)
        assert td["lambda_U"] > 0, "Gumbel lambda_U should be positive"

    def test_frank_tail_dependence_zero(
        self, synthetic_uniform_data: np.ndarray
    ) -> None:
        """Frank copula: lambda_L = lambda_U = 0."""
        params = frank_copula_fit(synthetic_uniform_data)
        td = tail_dependence(params)
        assert td["lambda_L"] == pytest.approx(0.0, abs=1e-12)
        assert td["lambda_U"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 8--9. Copula samples recover correct correlation
# ---------------------------------------------------------------------------

class TestSampling:

    def test_gaussian_sample_correlation(
        self, synthetic_uniform_data: np.ndarray, rho: float
    ) -> None:
        """Gaussian copula samples should recover the true correlation (within 0.05)."""
        params = gaussian_copula_fit(synthetic_uniform_data)
        u_sim = copula_sample(params, n_samples=10_000, seed=42)
        # Transform back to normal to check correlation
        z_sim = stats.norm.ppf(np.clip(u_sim, 1e-10, 1 - 1e-10))
        rho_hat = np.corrcoef(z_sim[:, 0], z_sim[:, 1])[0, 1]
        assert rho_hat == pytest.approx(rho, abs=0.05)

    def test_t_copula_sample_correlation(
        self, synthetic_uniform_data: np.ndarray, rho: float
    ) -> None:
        """t-copula samples should recover approximate correlation (within 0.05)."""
        params = t_copula_fit(synthetic_uniform_data)
        u_sim = copula_sample(params, n_samples=10_000, seed=42)
        z_sim = stats.norm.ppf(np.clip(u_sim, 1e-10, 1 - 1e-10))
        rho_hat = np.corrcoef(z_sim[:, 0], z_sim[:, 1])[0, 1]
        assert rho_hat == pytest.approx(rho, abs=0.05)

    def test_clayton_sample_unit_square(
        self, synthetic_uniform_data: np.ndarray
    ) -> None:
        """Clayton copula samples must lie in [0, 1]^2."""
        params = clayton_copula_fit(synthetic_uniform_data)
        u_sim = copula_sample(params, n_samples=5_000, seed=42)
        assert np.all(u_sim >= 0)
        assert np.all(u_sim <= 1)

    def test_copula_sample_shape(
        self, multivariate_uniform_data: np.ndarray
    ) -> None:
        """Copula samples should have shape (n_samples, d)."""
        params = gaussian_copula_fit(multivariate_uniform_data)
        n_samples = 8000
        u_sim = copula_sample(params, n_samples=n_samples, seed=42)
        assert u_sim.shape == (n_samples, 5)


# ---------------------------------------------------------------------------
# 11. VaR differs across copula families (synthetic comparison)
# ---------------------------------------------------------------------------

class TestVaRComparison:

    def test_var_differs_across_families(
        self, synthetic_uniform_data: np.ndarray
    ) -> None:
        """Portfolio VaR should differ across copula families.

        We simulate from each family, construct portfolio losses, and
        check that VaR values are not all identical.
        """
        families_params = {
            "gaussian": gaussian_copula_fit(synthetic_uniform_data),
            "t": t_copula_fit(synthetic_uniform_data),
            "clayton": clayton_copula_fit(synthetic_uniform_data),
            "gumbel": gumbel_copula_fit(synthetic_uniform_data),
            "frank": frank_copula_fit(synthetic_uniform_data),
        }

        alpha = 0.99
        var_values = []

        for family, params in families_params.items():
            u_sim = copula_sample(params, n_samples=10_000, seed=42)
            # Convert to "returns" via normal quantile
            z_sim = stats.norm.ppf(np.clip(u_sim, 1e-10, 1 - 1e-10))
            # Equal-weight portfolio
            weights = np.array([0.5, 0.5])
            port_returns = z_sim @ weights
            port_losses = -port_returns
            var_val = float(np.quantile(port_losses, alpha))
            var_values.append(var_val)

        # Not all VaR values should be the same
        assert len(set(round(v, 6) for v in var_values)) > 1, (
            "All copula families produced identical VaR -- expected variation."
        )
