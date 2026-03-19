"""Tests for Project 02: Monte Carlo Simulation Engine.

Covers:
  1. GBM expected value convergence: E[S_T] = S_0 * exp(mu * T)
  2. European call price converges to Black-Scholes within 2 std errors
  3. Antithetic variates reduce standard error vs naive MC
  4. Cholesky: simulated correlation matches input correlation within tolerance
  5. VaR monotonicity: VaR(0.99) > VaR(0.95)
"""

from __future__ import annotations

import numpy as np
import pytest

from risk_analyst.simulation.gbm import simulate_gbm, simulate_gbm_correlated
from risk_analyst.simulation.option_pricing import bs_price, price_european_option
from risk_analyst.simulation.risk import mc_portfolio_es, mc_portfolio_var

# -----------------------------------------------------------------------
# 1. GBM expected value: E[S_T] = S_0 * exp(mu * T)
# -----------------------------------------------------------------------

class TestGBMExpectedValue:
    """Test that GBM simulation mean converges to the analytic expectation."""

    def test_gbm_mean_convergence(self) -> None:
        s0 = 100.0
        mu = 0.08
        sigma = 0.2
        T = 1.0
        n_steps = 252
        n_paths = 200_000
        seed = 42

        paths = simulate_gbm(s0, mu, sigma, T, n_steps, n_paths, seed)

        # E[S_T] = S_0 * exp(mu * T)
        expected = s0 * np.exp(mu * T)
        simulated_mean = float(np.mean(paths[:, -1]))

        # Check within 2 % of analytic value
        rel_error = abs(simulated_mean - expected) / expected
        assert rel_error < 0.02, (
            f"GBM mean {simulated_mean:.4f} deviates from "
            f"E[S_T]={expected:.4f} by {rel_error:.4%}"
        )

    def test_gbm_initial_price(self) -> None:
        """All paths should start at s0."""
        s0 = 50.0
        paths = simulate_gbm(s0, 0.05, 0.3, 1.0, 100, 500, seed=7)
        np.testing.assert_allclose(paths[:, 0], s0)

    def test_gbm_shape(self) -> None:
        """Output shape must be (n_paths, n_steps + 1)."""
        paths = simulate_gbm(100, 0.05, 0.2, 1.0, 50, 1000, seed=1)
        assert paths.shape == (1000, 51)

    def test_gbm_positive_prices(self) -> None:
        """GBM prices must always be positive (log-normal property)."""
        paths = simulate_gbm(100, -0.1, 0.5, 2.0, 500, 5000, seed=99)
        assert np.all(paths > 0)


# -----------------------------------------------------------------------
# 2. European call price vs Black-Scholes
# -----------------------------------------------------------------------

class TestEuropeanOptionPricing:
    """MC European call should converge to BS within 2 standard errors."""

    @pytest.mark.parametrize(
        "s0, K, r, sigma, T",
        [
            (100, 100, 0.05, 0.2, 1.0),   # ATM
            (100, 110, 0.05, 0.2, 1.0),   # OTM call
            (100, 90, 0.05, 0.2, 1.0),    # ITM call
            (100, 100, 0.05, 0.4, 0.5),   # High vol, short maturity
        ],
    )
    def test_european_call_vs_bs(
        self, s0: float, K: float, r: float, sigma: float, T: float
    ) -> None:
        mc_price, mc_se, mc_ci = price_european_option(
            s0=s0, K=K, r=r, sigma=sigma, T=T,
            option_type="call", n_paths=500_000, seed=42,
        )
        analytical = bs_price(s0, K, r, sigma, T, "call")

        # MC price should be within 2 standard errors of BS
        assert abs(mc_price - analytical) < 2 * mc_se, (
            f"MC={mc_price:.4f}, BS={analytical:.4f}, SE={mc_se:.4f}, "
            f"diff={abs(mc_price - analytical):.4f} > 2*SE={2*mc_se:.4f}"
        )

    def test_european_put_vs_bs(self) -> None:
        s0, K, r, sigma, T = 100, 105, 0.03, 0.25, 0.75
        mc_price, mc_se, _ = price_european_option(
            s0=s0, K=K, r=r, sigma=sigma, T=T,
            option_type="put", n_paths=500_000, seed=123,
        )
        analytical = bs_price(s0, K, r, sigma, T, "put")
        assert abs(mc_price - analytical) < 2 * mc_se

    def test_bs_put_call_parity(self) -> None:
        """C - P = S_0 - K * exp(-r*T)."""
        s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        c = bs_price(s0, K, r, sigma, T, "call")
        p = bs_price(s0, K, r, sigma, T, "put")
        parity_rhs = s0 - K * np.exp(-r * T)
        assert abs((c - p) - parity_rhs) < 1e-10


# -----------------------------------------------------------------------
# 3. Antithetic variates reduce standard error
# -----------------------------------------------------------------------

class TestAntitheticVariates:
    """Antithetic variates should produce lower SE than naive MC."""

    def test_antithetic_reduces_se(self) -> None:
        s0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        n_paths = 50_000
        seed = 42

        # --- Naive MC ---
        _, naive_se, _ = price_european_option(
            s0=s0, K=K, r=r, sigma=sigma, T=T,
            option_type="call", n_paths=n_paths, seed=seed,
        )

        # --- Antithetic MC ---
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_paths)
        discount = np.exp(-r * T)

        s_T_pos = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        s_T_neg = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-z))

        payoff_avg = (np.maximum(s_T_pos - K, 0) + np.maximum(s_T_neg - K, 0)) / 2.0
        anti_discounted = discount * payoff_avg
        anti_se = float(np.std(anti_discounted, ddof=1) / np.sqrt(n_paths))

        # Antithetic SE should be strictly lower
        assert anti_se < naive_se, (
            f"Antithetic SE ({anti_se:.6f}) should be < naive SE ({naive_se:.6f})"
        )


# -----------------------------------------------------------------------
# 4. Cholesky correlation recovery
# -----------------------------------------------------------------------

class TestCholeskyCorrelation:
    """Simulated returns should recover the input correlation matrix."""

    def test_correlation_recovery(self) -> None:
        n_assets = 3
        s0_vec = np.array([100.0, 100.0, 100.0])
        mu_vec = np.array([0.05, 0.08, 0.06])
        sigma_vec = np.array([0.2, 0.3, 0.25])

        # Target correlation
        target_corr = np.array([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ])

        paths = simulate_gbm_correlated(
            s0_vec=s0_vec,
            mu_vec=mu_vec,
            sigma_vec=sigma_vec,
            corr_matrix=target_corr,
            T=1.0,
            n_steps=252,
            n_paths=100_000,
            seed=42,
        )

        # Compute simulated log-returns
        log_returns = np.log(paths[:, 1:, :] / paths[:, :-1, :])
        # Flatten steps: (n_paths * n_steps, n_assets)
        flat = log_returns.reshape(-1, n_assets)
        sim_corr = np.corrcoef(flat, rowvar=False)

        # Each element should be within 0.05 of target
        np.testing.assert_allclose(sim_corr, target_corr, atol=0.05)

    def test_correlated_output_shape(self) -> None:
        paths = simulate_gbm_correlated(
            s0_vec=np.array([100.0, 200.0]),
            mu_vec=np.array([0.05, 0.07]),
            sigma_vec=np.array([0.2, 0.3]),
            corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=1.0,
            n_steps=50,
            n_paths=1000,
            seed=1,
        )
        assert paths.shape == (1000, 51, 2)

    def test_correlated_initial_prices(self) -> None:
        s0 = np.array([50.0, 75.0, 120.0])
        paths = simulate_gbm_correlated(
            s0_vec=s0,
            mu_vec=np.array([0.05, 0.05, 0.05]),
            sigma_vec=np.array([0.2, 0.2, 0.2]),
            corr_matrix=np.eye(3),
            T=1.0,
            n_steps=10,
            n_paths=100,
            seed=5,
        )
        # Every path should start at the initial prices
        for i in range(len(s0)):
            np.testing.assert_allclose(paths[:, 0, i], s0[i])


# -----------------------------------------------------------------------
# 5. VaR monotonicity: VaR(0.99) > VaR(0.95)
# -----------------------------------------------------------------------

class TestVaRMonotonicity:
    """Higher confidence level should yield higher VaR."""

    def test_var_monotonicity(self) -> None:
        rng = np.random.default_rng(42)
        # Synthetic returns: 2 assets, 500 observations
        returns = rng.normal(0.0005, 0.02, size=(500, 2))
        weights = np.array([0.6, 0.4])

        var_95 = mc_portfolio_var(returns, weights, alpha=0.95, n_sims=50_000, seed=42)
        var_99 = mc_portfolio_var(returns, weights, alpha=0.99, n_sims=50_000, seed=42)

        assert var_99 > var_95, (
            f"VaR(0.99)={var_99:.6f} should be > VaR(0.95)={var_95:.6f}"
        )

    def test_es_exceeds_var(self) -> None:
        """ES >= VaR by definition (ES is the tail average)."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.015, size=(500, 3))
        weights = np.array([0.4, 0.3, 0.3])
        alpha = 0.95

        var = mc_portfolio_var(returns, weights, alpha=alpha, n_sims=50_000, seed=7)
        es = mc_portfolio_es(returns, weights, alpha=alpha, n_sims=50_000, seed=7)

        assert es >= var - 1e-10, (
            f"ES={es:.6f} should be >= VaR={var:.6f}"
        )

    def test_var_positive_for_normal_returns(self) -> None:
        """VaR should be positive for zero-mean returns at high confidence."""
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0, 0.02, size=(1000, 1))
        weights = np.array([1.0])

        var = mc_portfolio_var(returns, weights, alpha=0.95, n_sims=50_000, seed=99)
        assert var > 0, f"VaR should be positive, got {var:.6f}"
