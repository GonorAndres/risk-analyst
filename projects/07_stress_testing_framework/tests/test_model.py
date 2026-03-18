"""Tests for the stress testing framework.

Uses synthetic macro data and portfolio returns to validate every
major component: transmission model, DFAST scenarios, historical
replays, credit migration, reverse stress, and report generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transmission import MacroTransmissionModel, stress_transition_matrix, portfolio_loss_under_migration
from scenarios import get_dfast_scenarios, get_historical_scenarios, generate_stochastic_scenarios
from reverse_stress import reverse_stress_test
from model import StressTestFramework


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_data():
    """Generate reproducible synthetic macro factors and portfolio returns."""
    rng = np.random.default_rng(42)
    n = 40
    macro_factors = pd.DataFrame({
        "gdp_growth": rng.normal(0.02, 0.01, n),
        "unemployment": rng.normal(0.05, 0.01, n),
        "equity_index": rng.normal(0.02, 0.05, n),
        "interest_rate_10y": rng.normal(0.03, 0.005, n),
        "credit_spread": rng.normal(0.02, 0.005, n),
        "house_price_index": rng.normal(0.01, 0.02, n),
    })
    betas = np.array([0.5, -0.3, 0.8, -0.2, -0.6, 0.3])
    portfolio_returns = (macro_factors.values @ betas) + rng.normal(0, 0.01, n)
    return macro_factors, portfolio_returns, betas


@pytest.fixture()
def fitted_model(synthetic_data):
    """Return a MacroTransmissionModel already fitted on synthetic data."""
    macro_factors, portfolio_returns, _ = synthetic_data
    model = MacroTransmissionModel()
    model.fit(portfolio_returns, macro_factors)
    return model


@pytest.fixture()
def framework(synthetic_data):
    """Return a StressTestFramework with default config."""
    config = {
        "transmission": {
            "factors": [
                "gdp_growth", "unemployment", "equity_index",
                "interest_rate_10y", "credit_spread", "house_price_index",
            ],
        },
        "capital": {"initial_ratio": 0.12},
        "credit": {"lgd": 0.45, "stress_factor": 2.0},
        "reverse_stress": {"loss_threshold": 0.15},
    }
    return StressTestFramework(config)


# ---------------------------------------------------------------------------
# 1. Transmission model fits (R-squared reasonable)
# ---------------------------------------------------------------------------

def test_transmission_fit_r_squared(fitted_model):
    """R-squared should be high on synthetic data with known linear structure."""
    assert fitted_model.r_squared > 0.80, (
        f"R-squared too low: {fitted_model.r_squared:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Predicted loss has correct sign for adverse shocks
# ---------------------------------------------------------------------------

def test_predicted_loss_sign(fitted_model):
    """An adverse shock (negative GDP, rising unemployment) should produce positive loss."""
    adverse_shocks = {
        "gdp_growth": -0.04,
        "unemployment": 0.10,
        "equity_index": -0.30,
        "interest_rate_10y": -0.02,
        "credit_spread": 0.05,
        "house_price_index": -0.10,
    }
    loss = fitted_model.predict_loss(adverse_shocks)
    assert loss > 0, f"Expected positive loss for adverse scenario, got {loss:.6f}"


# ---------------------------------------------------------------------------
# 3. Sensitivity table has correct columns
# ---------------------------------------------------------------------------

def test_sensitivity_table_columns(fitted_model):
    """Sensitivity table must have factor, beta, t_stat, p_value columns."""
    table = fitted_model.sensitivity_table()
    required_cols = {"factor", "beta", "t_stat", "p_value"}
    assert required_cols.issubset(set(table.columns)), (
        f"Missing columns: {required_cols - set(table.columns)}"
    )
    assert len(table) == 6, f"Expected 6 factors, got {len(table)}"


# ---------------------------------------------------------------------------
# 4. DFAST scenarios exist (baseline, adverse, severely_adverse)
# ---------------------------------------------------------------------------

def test_dfast_scenarios_keys():
    """get_dfast_scenarios must return baseline, adverse, severely_adverse."""
    scenarios = get_dfast_scenarios()
    expected = {"baseline", "adverse", "severely_adverse"}
    assert set(scenarios.keys()) == expected


# ---------------------------------------------------------------------------
# 5. Severely adverse loss > adverse loss > baseline loss
# ---------------------------------------------------------------------------

def test_dfast_loss_ordering(framework, synthetic_data):
    """Loss severity must increase: baseline < adverse < severely_adverse."""
    macro_factors, portfolio_returns, _ = synthetic_data
    results = framework.run_dfast(portfolio_returns, macro_factors)

    losses = results.set_index("scenario")["cumulative_loss"]
    assert losses["severely_adverse"] > losses["adverse"], (
        "Severely adverse should produce higher loss than adverse"
    )
    assert losses["adverse"] > losses["baseline"], (
        "Adverse should produce higher loss than baseline"
    )


# ---------------------------------------------------------------------------
# 6. Historical scenarios have expected keys
# ---------------------------------------------------------------------------

def test_historical_scenarios_keys():
    """get_historical_scenarios must return gfc_2008, covid_2020, svb_2023."""
    scenarios = get_historical_scenarios()
    expected = {"gfc_2008", "covid_2020", "svb_2023"}
    assert set(scenarios.keys()) == expected


# ---------------------------------------------------------------------------
# 7. Stressed transition matrix rows sum to 1
# ---------------------------------------------------------------------------

def test_stressed_transition_matrix_rows_sum():
    """Each row of the stressed transition matrix must sum to 1."""
    base_matrix = np.array([
        [0.90, 0.07, 0.02, 0.005, 0.005],
        [0.05, 0.85, 0.07, 0.02, 0.01],
        [0.01, 0.05, 0.84, 0.07, 0.03],
        [0.005, 0.02, 0.05, 0.82, 0.105],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ])
    stressed = stress_transition_matrix(base_matrix, stress_factor=2.0)
    row_sums = stressed.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
                               err_msg="Stressed matrix rows do not sum to 1")


# ---------------------------------------------------------------------------
# 8. Stressed transition matrix default probabilities >= base
# ---------------------------------------------------------------------------

def test_stressed_default_probs_increase():
    """Stressed default probabilities should be >= base default probabilities."""
    base_matrix = np.array([
        [0.90, 0.07, 0.02, 0.005, 0.005],
        [0.05, 0.85, 0.07, 0.02, 0.01],
        [0.01, 0.05, 0.84, 0.07, 0.03],
        [0.005, 0.02, 0.05, 0.82, 0.105],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ])
    stressed = stress_transition_matrix(base_matrix, stress_factor=2.0)
    default_col = base_matrix.shape[1] - 1

    for i in range(base_matrix.shape[0] - 1):  # skip default state itself
        assert stressed[i, default_col] >= base_matrix[i, default_col] - 1e-12, (
            f"Row {i}: stressed PD {stressed[i, default_col]:.6f} "
            f"< base PD {base_matrix[i, default_col]:.6f}"
        )


# ---------------------------------------------------------------------------
# 9. Reverse stress test returns a result dict with expected keys
# ---------------------------------------------------------------------------

def test_reverse_stress_result_keys(fitted_model):
    """reverse_stress_test result must contain expected keys."""
    factor_names = fitted_model.factor_names
    result = reverse_stress_test(fitted_model, loss_threshold=0.10, factor_names=factor_names)
    expected_keys = {"optimal_shocks", "predicted_loss", "shock_norm", "success"}
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 10. Reverse stress predicted loss >= threshold (or close)
# ---------------------------------------------------------------------------

def test_reverse_stress_meets_threshold(fitted_model):
    """Reverse stress test should find a scenario meeting (or nearly meeting) the threshold."""
    threshold = 0.10
    factor_names = fitted_model.factor_names
    result = reverse_stress_test(fitted_model, loss_threshold=threshold, factor_names=factor_names)
    assert result["predicted_loss"] >= threshold * 0.95, (
        f"Predicted loss {result['predicted_loss']:.4f} too far below threshold {threshold}"
    )


# ---------------------------------------------------------------------------
# 11. Stochastic scenarios have correct shape
# ---------------------------------------------------------------------------

def test_stochastic_scenarios_shape():
    """generate_stochastic_scenarios output shape must be (n_scenarios, k)."""
    n_scenarios = 500
    k = 6
    means = np.zeros(k)
    cov = np.eye(k) * 0.01
    scenarios = generate_stochastic_scenarios(n_scenarios, means, cov, seed=42)
    assert scenarios.shape == (n_scenarios, k), (
        f"Expected shape ({n_scenarios}, {k}), got {scenarios.shape}"
    )


# ---------------------------------------------------------------------------
# 12. Portfolio loss under migration is non-negative
# ---------------------------------------------------------------------------

def test_portfolio_loss_under_migration_nonnegative():
    """Expected loss from credit migration must be >= 0."""
    base_matrix = np.array([
        [0.90, 0.07, 0.02, 0.005, 0.005],
        [0.05, 0.85, 0.07, 0.02, 0.01],
        [0.01, 0.05, 0.84, 0.07, 0.03],
        [0.005, 0.02, 0.05, 0.82, 0.105],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ])
    exposures = np.array([1_000_000, 500_000, 750_000, 300_000])
    ratings = np.array([0, 1, 2, 3])
    lgd = 0.45

    loss = portfolio_loss_under_migration(exposures, ratings, base_matrix, lgd)
    assert loss >= 0, f"Portfolio loss should be non-negative, got {loss:.2f}"


# ---------------------------------------------------------------------------
# 13. Report is non-empty string
# ---------------------------------------------------------------------------

def test_report_nonempty(framework, synthetic_data):
    """Generated report should be a non-empty markdown string."""
    macro_factors, portfolio_returns, _ = synthetic_data
    framework.run_dfast(portfolio_returns, macro_factors)
    framework.run_historical(portfolio_returns, macro_factors)
    framework.run_reverse(portfolio_returns, macro_factors, threshold=0.10)

    report = framework.generate_report()
    assert isinstance(report, str)
    assert len(report) > 100, "Report seems too short"
    assert "# Stress Test Report" in report


# ---------------------------------------------------------------------------
# 14. Scenario paths DataFrame has correct shape
# ---------------------------------------------------------------------------

def test_scenario_paths_shape(fitted_model):
    """predict_path should return a Series with length equal to scenario quarters."""
    scenarios = get_dfast_scenarios()
    baseline_df = scenarios["baseline"]

    path = fitted_model.predict_path(baseline_df)
    assert isinstance(path, pd.Series)
    assert len(path) == len(baseline_df), (
        f"Path length {len(path)} != scenario quarters {len(baseline_df)}"
    )
