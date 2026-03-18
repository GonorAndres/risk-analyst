"""Tests for EVT tail risk analysis.

Uses synthetic heavy-tailed data (Pareto, Frechet, exponential) -- no
yfinance downloads.  All tests are deterministic with fixed random seeds.

References:
    - Pareto(alpha=3) has xi = 1/alpha = 0.333
    - Frechet(alpha) <=> GEV with xi = 1/alpha
    - Exponential(1) <=> GPD with xi = 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_analyst.models.evt import (
    evt_es,
    evt_var,
    fit_gev,
    fit_gpd,
    return_level,
)

# Project-local imports (conftest.py adds project src/ to sys.path)
from threshold import (
    hill_estimator,
    mean_residual_life,
    parameter_stability,
    select_threshold_auto,
)
from model import EVTModel


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def pareto_data() -> np.ndarray:
    """Pareto(alpha=3) data: xi = 1/alpha = 0.333."""
    rng = np.random.default_rng(42)
    return rng.pareto(3, size=5000)


@pytest.fixture
def exponential_data() -> np.ndarray:
    """Exponential(1) data: xi = 0 (light-tailed)."""
    rng = np.random.default_rng(42)
    return rng.exponential(1.0, size=5000)


@pytest.fixture
def frechet_data() -> np.ndarray:
    """Frechet(alpha=2) data via inverse transform: xi = 1/alpha = 0.5."""
    rng = np.random.default_rng(42)
    # Frechet(alpha=2): F(x) = exp(-x^{-2}), use inverse transform
    # X = (-log(U))^{-1/2} for U ~ Uniform(0,1)
    u = rng.uniform(0, 1, size=2000)
    return (-np.log(u)) ** (-0.5)


@pytest.fixture
def default_config() -> dict:
    """Minimal config for EVTModel."""
    return {
        "pot": {
            "threshold_percentile": 95.0,
            "threshold_method": "percentile",
        },
        "block_maxima": {
            "block_size": 21,
        },
        "risk": {
            "confidence_levels": [0.95, 0.99, 0.995, 0.999],
        },
    }


# ---------------------------------------------------------------------------
# Test 1: GPD xi > 0 on synthetic heavy-tailed data (Pareto)
# ---------------------------------------------------------------------------

def test_gpd_positive_xi_pareto(pareto_data: np.ndarray) -> None:
    """GPD fitted to Pareto exceedances should yield xi > 0."""
    threshold = float(np.percentile(pareto_data, 90))
    exceedances = pareto_data[pareto_data > threshold] - threshold

    result = fit_gpd(exceedances, threshold)
    assert result["xi"] > 0, f"Expected xi > 0, got {result['xi']}"


# ---------------------------------------------------------------------------
# Test 2: GPD xi approx 0 on exponential data
# ---------------------------------------------------------------------------

def test_gpd_xi_near_zero_exponential(exponential_data: np.ndarray) -> None:
    """GPD fitted to exponential exceedances should yield xi close to 0."""
    threshold = float(np.percentile(exponential_data, 90))
    exceedances = exponential_data[exponential_data > threshold] - threshold

    result = fit_gpd(exceedances, threshold)
    assert abs(result["xi"]) < 0.15, (
        f"Expected xi near 0 for exponential data, got {result['xi']}"
    )


# ---------------------------------------------------------------------------
# Test 3: EVT VaR(0.999) > normal VaR(0.999) for heavy-tailed data
# ---------------------------------------------------------------------------

def test_evt_var_exceeds_normal_var(
    pareto_data: np.ndarray, default_config: dict
) -> None:
    """For heavy-tailed data, EVT VaR should exceed normal VaR at 99.9%."""
    model = EVTModel(default_config)
    model.fit_pot(pareto_data)
    risk = model.compute_risk(0.999)

    mu = float(np.mean(pareto_data))
    sigma = float(np.std(pareto_data, ddof=1))
    var_normal = mu + sigma * stats.norm.ppf(0.999)

    assert risk["var_evt"] > var_normal, (
        f"EVT VaR ({risk['var_evt']:.4f}) should exceed normal VaR "
        f"({var_normal:.4f}) for heavy-tailed data."
    )


# ---------------------------------------------------------------------------
# Test 4: EVT ES >= EVT VaR always
# ---------------------------------------------------------------------------

def test_evt_es_geq_var(
    pareto_data: np.ndarray, default_config: dict
) -> None:
    """Expected Shortfall must always be >= VaR (subadditivity property)."""
    model = EVTModel(default_config)
    model.fit_pot(pareto_data)

    for alpha in [0.95, 0.99, 0.995, 0.999]:
        risk = model.compute_risk(alpha)
        assert risk["es_evt"] >= risk["var_evt"], (
            f"ES ({risk['es_evt']:.4f}) < VaR ({risk['var_evt']:.4f}) "
            f"at alpha={alpha}"
        )


# ---------------------------------------------------------------------------
# Test 5: Return level increases with return period (monotonicity)
# ---------------------------------------------------------------------------

def test_return_level_monotonic(
    pareto_data: np.ndarray, default_config: dict
) -> None:
    """Return levels should be monotonically increasing in return period."""
    model = EVTModel(default_config)
    model.fit_block_maxima(pareto_data, block_size=21)

    periods = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    df = model.return_levels(periods)

    levels = df["return_level"].values
    for i in range(len(levels) - 1):
        assert levels[i + 1] > levels[i], (
            f"Return level at T={periods[i+1]} ({levels[i+1]:.4f}) "
            f"should exceed T={periods[i]} ({levels[i]:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 6: GEV fit on synthetic Frechet data: recovered xi within 0.15
# ---------------------------------------------------------------------------

def test_gev_xi_recovery_frechet(frechet_data: np.ndarray) -> None:
    """GEV fit on Frechet(alpha=2) block maxima should recover xi ~ 0.5."""
    # Take block maxima of size 50
    n_blocks = len(frechet_data) // 50
    trimmed = frechet_data[: n_blocks * 50]
    blocks = trimmed.reshape(n_blocks, 50)
    block_max = blocks.max(axis=1)

    result = fit_gev(block_max)
    expected_xi = 0.5
    assert abs(result["xi"] - expected_xi) < 0.15, (
        f"Expected xi ~ {expected_xi}, got {result['xi']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: Mean residual life is approximately linear above correct threshold
# ---------------------------------------------------------------------------

def test_mrl_linearity_pareto(pareto_data: np.ndarray) -> None:
    """For Pareto data, MRL should be approximately linear across thresholds."""
    thresholds = np.linspace(
        np.percentile(pareto_data, 50),
        np.percentile(pareto_data, 95),
        20,
    )
    thresh_out, mrl = mean_residual_life(pareto_data, thresholds)

    # Fit linear regression; R^2 should be high for Pareto
    if len(thresh_out) >= 5:
        slope, intercept, r_value, _, _ = stats.linregress(thresh_out, mrl)
        r_squared = r_value ** 2
        assert r_squared > 0.8, (
            f"MRL should be approximately linear for Pareto data, "
            f"got R^2 = {r_squared:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 8: Hill estimator converges to true tail index for Pareto data
# ---------------------------------------------------------------------------

def test_hill_estimator_pareto(pareto_data: np.ndarray) -> None:
    """Hill estimator should converge to xi = 1/alpha = 1/3 for Pareto(3)."""
    k_range = np.arange(50, 500, 10)
    k_values, hill_est = hill_estimator(pareto_data, k_range)

    # Take the median estimate in the stable region (k=100 to k=300)
    # Tolerance 0.15 accounts for finite-sample bias of Hill estimator
    mask = (k_values >= 100) & (k_values <= 300)
    if np.sum(mask) > 0:
        median_est = float(np.median(hill_est[mask]))
        expected_xi = 1.0 / 3.0
        assert abs(median_est - expected_xi) < 0.15, (
            f"Hill estimator median ({median_est:.4f}) should be close to "
            f"xi = {expected_xi:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 9: EVT VaR is finite and positive for valid alphas
# ---------------------------------------------------------------------------

def test_evt_var_finite_positive(pareto_data: np.ndarray) -> None:
    """EVT VaR should be finite and positive for reasonable alphas."""
    threshold = float(np.percentile(pareto_data, 95))
    exceedances = pareto_data[pareto_data > threshold] - threshold
    gpd_params = fit_gpd(exceedances, threshold)
    n_total = len(pareto_data)

    for alpha in [0.95, 0.99, 0.995, 0.999]:
        var = evt_var(gpd_params, n_total, alpha)
        assert np.isfinite(var), f"VaR not finite at alpha={alpha}"
        assert var > 0, f"VaR not positive at alpha={alpha}: {var}"


# ---------------------------------------------------------------------------
# Test 10: Block maxima count equals n_observations / block_size
# ---------------------------------------------------------------------------

def test_block_maxima_count(
    pareto_data: np.ndarray, default_config: dict
) -> None:
    """Number of block maxima should be floor(n / block_size)."""
    model = EVTModel(default_config)
    block_size = 21
    result = model.fit_block_maxima(pareto_data, block_size=block_size)

    expected_n_blocks = len(pareto_data) // block_size
    assert result["n_blocks"] == expected_n_blocks, (
        f"Expected {expected_n_blocks} blocks, got {result['n_blocks']}"
    )


# ---------------------------------------------------------------------------
# Test 11: GPD exceedances are all positive
# ---------------------------------------------------------------------------

def test_gpd_exceedances_positive(pareto_data: np.ndarray) -> None:
    """Exceedances passed to fit_gpd should all be positive."""
    threshold = float(np.percentile(pareto_data, 90))
    exceedances = pareto_data[pareto_data > threshold] - threshold

    assert np.all(exceedances > 0), "All exceedances must be positive."

    # Also verify fit_gpd raises on non-positive input
    with pytest.raises(ValueError, match="strictly positive"):
        fit_gpd(np.array([0.0, 1.0, 2.0]), threshold=0.0)


# ---------------------------------------------------------------------------
# Test 12: compare_methods returns correct DataFrame shape
# ---------------------------------------------------------------------------

def test_compare_methods_shape(
    pareto_data: np.ndarray, default_config: dict
) -> None:
    """compare_methods should return a DataFrame with correct columns and rows."""
    model = EVTModel(default_config)
    model.fit_pot(pareto_data)

    alphas = [0.95, 0.99, 0.995, 0.999]
    df = model.compare_methods(pareto_data, alphas)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(alphas)

    expected_cols = {
        "alpha", "var_normal", "var_t", "var_evt",
        "es_normal", "es_t", "es_evt",
    }
    assert set(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {set(df.columns)}"
    )

    # All risk measures should be finite
    for col in expected_cols - {"alpha"}:
        assert df[col].apply(np.isfinite).all(), (
            f"Column '{col}' contains non-finite values."
        )
