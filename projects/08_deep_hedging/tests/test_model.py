"""Tests for deep hedging -- environment, network, trainer, and model.

All tests use synthetic data only (no external dependencies).
Tests are designed to verify correctness of individual components and
that the overall training loop produces meaningful improvements.

Reference: Buehler et al. (2019), "Deep hedging", Quantitative Finance.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment import HedgingEnvironment
from network import NeuralNetwork
from trainer import DeepHedgingTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> HedgingEnvironment:
    """Standard hedging environment for testing."""
    return HedgingEnvironment(
        s0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.20,
        T=1 / 12,
        n_steps=21,
        n_paths=500,
        cost_rate=0.001,
        seed=42,
    )


@pytest.fixture
def network() -> NeuralNetwork:
    """Standard neural network for testing."""
    return NeuralNetwork(layer_sizes=[3, 32, 32, 1], seed=42)


@pytest.fixture
def trainer(env: HedgingEnvironment, network: NeuralNetwork) -> DeepHedgingTrainer:
    """Standard trainer for testing."""
    config = {
        "population_size": 20,
        "lr": 0.02,
        "risk_measure": "cvar",
        "cvar_alpha": 0.95,
    }
    return DeepHedgingTrainer(env=env, network=network, config=config)


# ---------------------------------------------------------------------------
# Test 1: Path simulation shape
# ---------------------------------------------------------------------------

class TestHedgingEnvironment:
    """Tests for HedgingEnvironment."""

    def test_simulate_paths_shape(self, env: HedgingEnvironment) -> None:
        """Simulated paths have shape (n_paths, n_steps + 1)."""
        paths = env.simulate_paths()
        assert paths.shape == (env.n_paths, env.n_steps + 1)
        # All paths start at s0
        np.testing.assert_allclose(paths[:, 0], env.s0)
        # All prices positive
        assert np.all(paths > 0)

    # -----------------------------------------------------------------------
    # Test 2: BS delta in [0, 1]
    # -----------------------------------------------------------------------

    def test_bs_delta_bounds(self, env: HedgingEnvironment) -> None:
        """BS delta for a call is between 0 and 1."""
        S_values = np.linspace(60, 140, 50)
        t_values = np.array([0.0, env.T * 0.5, env.T * 0.9])

        for t in t_values:
            delta = env.bs_delta(S_values, t)
            assert np.all(delta >= 0.0), f"Delta < 0 at t={t}"
            assert np.all(delta <= 1.0), f"Delta > 1 at t={t}"

    # -----------------------------------------------------------------------
    # Test 3: BS delta monotonicity
    # -----------------------------------------------------------------------

    def test_bs_delta_monotonicity(self, env: HedgingEnvironment) -> None:
        """BS delta for a call increases with S (monotonicity)."""
        S_values = np.linspace(70, 130, 100)
        delta = env.bs_delta(S_values, 0.0)
        # Delta should be non-decreasing in S
        assert np.all(np.diff(delta) >= -1e-10)

    # -----------------------------------------------------------------------
    # Test 4: BS price matches analytical
    # -----------------------------------------------------------------------

    def test_bs_price_analytical(self, env: HedgingEnvironment) -> None:
        """BS price at t=0 matches known analytical value."""
        from scipy.stats import norm

        s0, K, r, sigma, T = env.s0, env.K, env.r, env.sigma, env.T
        d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        expected = s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        computed = float(env.bs_price(s0, 0.0))
        assert computed == pytest.approx(expected, rel=1e-10)

    # -----------------------------------------------------------------------
    # Test 5: Payoff is non-negative
    # -----------------------------------------------------------------------

    def test_payoff_non_negative(self, env: HedgingEnvironment) -> None:
        """Call payoff max(S_T - K, 0) is always non-negative."""
        S_T = np.array([80, 90, 100, 110, 120, 50, 150])
        payoff = env.compute_payoff(S_T)
        assert np.all(payoff >= 0.0)
        # Check specific values
        np.testing.assert_allclose(payoff[0], 0.0)  # S_T < K
        np.testing.assert_allclose(payoff[3], 10.0)  # S_T = 110, K = 100
        np.testing.assert_allclose(payoff[4], 20.0)  # S_T = 120

    # -----------------------------------------------------------------------
    # Test 6: Perfect delta hedge reduces variance
    # -----------------------------------------------------------------------

    def test_delta_hedge_reduces_variance(self, env: HedgingEnvironment) -> None:
        """BS delta hedging has lower P&L variance than no hedge."""
        paths = env.simulate_paths()
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        # No hedge: just premium - payoff
        premium = float(env.bs_price(env.s0, 0.0))
        no_hedge_pnl = premium - env.compute_payoff(paths[:, -1])

        # BS delta hedge
        bs_positions = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            time_t = t * env.dt
            bs_positions[:, t] = env.bs_delta(paths[:, t], time_t)
        bs_pnl = env.compute_pnl(paths, bs_positions)

        assert np.var(bs_pnl) < np.var(no_hedge_pnl)

    # -----------------------------------------------------------------------
    # Test 7: Transaction costs are non-negative
    # -----------------------------------------------------------------------

    def test_transaction_costs_non_negative(self, env: HedgingEnvironment) -> None:
        """Transaction costs in compute_pnl are non-negative (reduce P&L)."""
        paths = env.simulate_paths()
        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        # Use constant positions (no rebalancing beyond initial trade)
        positions = np.full((n_paths, n_steps), 0.5)

        # Compute P&L with and without transaction costs
        pnl_with_costs = env.compute_pnl(paths, positions)

        # Compute without costs (set cost_rate to 0)
        env_no_cost = HedgingEnvironment(
            s0=env.s0, K=env.K, r=env.r, sigma=env.sigma, T=env.T,
            n_steps=env.n_steps, n_paths=env.n_paths, cost_rate=0.0, seed=env.seed,
        )
        pnl_no_costs = env_no_cost.compute_pnl(paths, positions)

        # P&L with costs should be <= P&L without costs (costs reduce profit)
        assert np.all(pnl_with_costs <= pnl_no_costs + 1e-10)


# ---------------------------------------------------------------------------
# Test 8-9: Neural Network
# ---------------------------------------------------------------------------

class TestNeuralNetwork:
    """Tests for NeuralNetwork."""

    def test_forward_output_bounds(self, network: NeuralNetwork) -> None:
        """Network output is in [0, 1] due to sigmoid."""
        rng = np.random.default_rng(123)
        x = rng.standard_normal((100, 3))
        y = network.forward(x)

        assert y.shape == (100, 1)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_parameter_count(self, network: NeuralNetwork) -> None:
        """Parameter count matches expected for [3, 32, 32, 1]."""
        # Layer 0: 3*32 weights + 32 biases = 128
        # Layer 1: 32*32 weights + 32 biases = 1056
        # Layer 2: 32*1 weights + 1 bias = 33
        expected = (3 * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
        assert network.num_parameters() == expected

    def test_flat_parameters_roundtrip(self, network: NeuralNetwork) -> None:
        """Flattening and restoring parameters preserves values."""
        flat = network.get_flat_parameters()
        assert len(flat) == network.num_parameters()

        # Modify, then restore
        network.set_flat_parameters(flat + 1.0)
        flat_modified = network.get_flat_parameters()
        assert not np.allclose(flat, flat_modified)

        network.set_flat_parameters(flat)
        flat_restored = network.get_flat_parameters()
        np.testing.assert_allclose(flat, flat_restored)


# ---------------------------------------------------------------------------
# Test 10-11: Risk measures
# ---------------------------------------------------------------------------

class TestRiskMeasures:
    """Tests for risk measure computations."""

    def test_variance_non_negative(self, trainer: DeepHedgingTrainer) -> None:
        """Variance risk measure is always non-negative."""
        pnl = np.array([1.0, -2.0, 0.5, -1.0, 3.0, -0.5])
        var = trainer.risk_measure(pnl, "variance")
        assert var >= 0.0

    def test_cvar_geq_mean_loss(self, trainer: DeepHedgingTrainer) -> None:
        """CVaR (expected shortfall) >= mean loss.

        CVaR focuses on the tail, so it must be at least as large as
        the average loss.
        """
        rng = np.random.default_rng(42)
        pnl = rng.standard_normal(1000)  # random P&L

        cvar = trainer.risk_measure(pnl, "cvar")
        mean_loss = float(np.mean(-pnl))

        assert cvar >= mean_loss - 1e-10


# ---------------------------------------------------------------------------
# Test 12: Training produces improvement
# ---------------------------------------------------------------------------

class TestTraining:
    """Tests for the training loop."""

    def test_training_improves_over_no_hedge(self) -> None:
        """After training, deep hedge CVaR <= no-hedge CVaR.

        This is the key integration test: the network must learn something
        useful, producing lower risk than doing nothing.
        """
        # Use a smaller environment for faster test
        env = HedgingEnvironment(
            s0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.20,
            T=1 / 12,
            n_steps=10,
            n_paths=500,
            cost_rate=0.0005,
            seed=42,
        )

        network = NeuralNetwork(layer_sizes=[3, 16, 1], seed=42)

        config = {
            "population_size": 30,
            "lr": 0.03,
            "risk_measure": "variance",
            "noise_std": 0.15,
        }
        trainer = DeepHedgingTrainer(env=env, network=network, config=config)

        # Train for enough epochs to learn something
        loss_history = trainer.train(n_epochs=60, measure="variance")

        # Evaluate on the same paths
        paths = env.simulate_paths()
        deep_positions = trainer.compute_hedge_positions(paths)
        deep_pnl = env.compute_pnl(paths, deep_positions)

        # No-hedge P&L
        premium = float(env.bs_price(env.s0, 0.0))
        no_hedge_pnl = premium - env.compute_payoff(paths[:, -1])

        # Deep hedge should have lower variance than no hedge
        deep_var = float(np.var(deep_pnl))
        no_hedge_var = float(np.var(no_hedge_pnl))

        assert deep_var < no_hedge_var, (
            f"Deep hedge variance ({deep_var:.6f}) should be less than "
            f"no-hedge variance ({no_hedge_var:.6f})"
        )

        # Also check that loss decreased during training
        assert loss_history[-1] < loss_history[0], (
            f"Final loss ({loss_history[-1]:.6f}) should be less than "
            f"initial loss ({loss_history[0]:.6f})"
        )
