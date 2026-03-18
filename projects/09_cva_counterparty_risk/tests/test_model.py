"""Tests for CVA counterparty risk project.

14 tests covering Vasicek simulation, credit modeling, exposure
profiles, CVA computation, netting, collateral, and wrong-way risk.

All tests use synthetic Vasicek paths -- no external data.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit import (
    default_probability,
    hazard_rate_from_cds,
    marginal_default_prob,
    survival_probability,
)
from cva import compute_bilateral_cva, compute_cva, cva_by_netting_set
from exposure import apply_collateral, apply_netting, compute_exposure_profiles
from instruments import InterestRateSwap, simulate_rate_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
N_PATHS = 2000
N_STEPS = 20
R0 = 0.03
KAPPA = 0.5
THETA = 0.04
SIGMA = 0.01
T = 5.0


@pytest.fixture(scope="module")
def vasicek_paths():
    """Simulate Vasicek rate paths for reuse across tests."""
    paths, times = simulate_rate_paths(
        r0=R0, kappa=KAPPA, theta=THETA, sigma=SIGMA,
        T=T, n_steps=N_STEPS, n_paths=N_PATHS, seed=SEED,
    )
    return paths, times


@pytest.fixture(scope="module")
def swap_values(vasicek_paths):
    """Compute swap MTM values from Vasicek paths."""
    paths, times = vasicek_paths
    swap = InterestRateSwap(
        notional=1_000_000, fixed_rate=0.04, tenor=T,
        payment_freq=0.25, seed=SEED,
    )
    return swap.simulate_values(paths, times), times


@pytest.fixture(scope="module")
def profiles(swap_values):
    """Compute exposure profiles from swap values."""
    mtm, times = swap_values
    return compute_exposure_profiles(mtm, times), times


# ---------------------------------------------------------------------------
# Test 1: Vasicek paths mean-revert toward theta
# ---------------------------------------------------------------------------

def test_vasicek_mean_reversion(vasicek_paths):
    """Final mean of rate paths should be close to long-run mean theta."""
    paths, _ = vasicek_paths
    final_mean = np.mean(paths[:, -1])
    # With kappa=0.5 and T=5, mean should approach theta=0.04
    assert final_mean == pytest.approx(THETA, abs=0.005)


# ---------------------------------------------------------------------------
# Test 2: Vasicek paths have correct shape
# ---------------------------------------------------------------------------

def test_vasicek_shape(vasicek_paths):
    """Rate paths should have shape (n_paths, n_steps + 1)."""
    paths, times = vasicek_paths
    assert paths.shape == (N_PATHS, N_STEPS + 1)
    assert times.shape == (N_STEPS + 1,)
    assert paths[:, 0] == pytest.approx(R0)


# ---------------------------------------------------------------------------
# Test 3: Hazard rate from CDS: lambda = s / (1 - R) exactly
# ---------------------------------------------------------------------------

def test_hazard_rate_from_cds():
    """Hazard rate should equal spread / (1 - recovery) exactly."""
    spread = 0.01
    recovery = 0.40
    expected = spread / (1.0 - recovery)
    assert hazard_rate_from_cds(spread, recovery) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 4: Survival probability decreases with time
# ---------------------------------------------------------------------------

def test_survival_decreasing():
    """Survival probability must be monotonically decreasing in time."""
    h = 0.02
    times_test = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    surv = [survival_probability(h, t) for t in times_test]
    for i in range(len(surv) - 1):
        assert surv[i] > surv[i + 1]


# ---------------------------------------------------------------------------
# Test 5: Default probability = 1 - survival probability
# ---------------------------------------------------------------------------

def test_default_probability_identity():
    """PD(t) + Q(t) must equal 1 for all t."""
    h = 0.02
    for t in [0.5, 1.0, 2.0, 5.0]:
        pd = default_probability(h, t)
        q = survival_probability(h, t)
        assert pd + q == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6: Marginal default probability is non-negative
# ---------------------------------------------------------------------------

def test_marginal_default_prob_nonneg():
    """Marginal default probability must be non-negative for t2 > t1."""
    h = 0.02
    intervals = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]
    for t1, t2 in intervals:
        mpd = marginal_default_prob(h, t1, t2)
        assert mpd >= 0.0


# ---------------------------------------------------------------------------
# Test 7: EE is non-negative at all times
# ---------------------------------------------------------------------------

def test_ee_nonneg(profiles):
    """Expected Exposure must be non-negative everywhere."""
    prof, _ = profiles
    assert np.all(prof["ee"] >= -1e-12)


# ---------------------------------------------------------------------------
# Test 8: PFE >= EE at all times
# ---------------------------------------------------------------------------

def test_pfe_ge_ee(profiles):
    """PFE (97.5%) must be >= EE at every time point."""
    prof, _ = profiles
    # Use tolerance for floating-point percentile vs mean rounding at t=0
    assert np.all(prof["pfe_975"] >= prof["ee"] - 1e-6)


# ---------------------------------------------------------------------------
# Test 9: EPE is positive
# ---------------------------------------------------------------------------

def test_epe_positive(profiles):
    """Effective Expected Positive Exposure must be strictly positive."""
    prof, _ = profiles
    assert prof["epe"] > 0.0


# ---------------------------------------------------------------------------
# Test 10: CVA is positive
# ---------------------------------------------------------------------------

def test_cva_positive(profiles):
    """CVA should be positive when there is positive expected exposure."""
    prof, times = profiles
    hazard = hazard_rate_from_cds(0.01, 0.40)
    cva = compute_cva(prof["ee"], times, hazard, 0.40)
    assert cva > 0.0


# ---------------------------------------------------------------------------
# Test 11: Netting reduces CVA vs gross
# ---------------------------------------------------------------------------

def test_netting_reduces_cva(vasicek_paths):
    """Netting multiple trades should reduce CVA vs gross sum."""
    paths, times = vasicek_paths
    hazard = hazard_rate_from_cds(0.01, 0.40)

    # Create 3 swaps with different fixed rates
    trades_mtm = []
    for i, offset in enumerate([-0.005, 0.0, 0.005]):
        swap = InterestRateSwap(
            notional=1_000_000, fixed_rate=0.04 + offset,
            tenor=T, payment_freq=0.25, seed=SEED + i,
        )
        mtm = swap.simulate_values(paths, times)
        trades_mtm.append(mtm)

    result = cva_by_netting_set(trades_mtm, times, hazard, 0.40)
    assert result["net_cva"] <= result["gross_cva"] + 1e-12


# ---------------------------------------------------------------------------
# Test 12: Collateral reduces exposure
# ---------------------------------------------------------------------------

def test_collateral_reduces_exposure(swap_values):
    """Collateral with a threshold should reduce exposure."""
    mtm, _ = swap_values
    positive_exp = np.maximum(mtm, 0.0)

    # Apply collateral with threshold = 50000, MTA = 1000
    collateralized = apply_collateral(mtm, threshold=50000.0, mta=1000.0)

    # Collateralized mean exposure should be lower
    mean_uncoll = np.mean(positive_exp)
    mean_coll = np.mean(collateralized)
    assert mean_coll <= mean_uncoll + 1e-12


# ---------------------------------------------------------------------------
# Test 13: Wrong-way risk increases CVA with positive correlation
# ---------------------------------------------------------------------------

def test_wrong_way_risk(vasicek_paths):
    """CVA should increase with positive exposure-default correlation."""
    paths, times = vasicek_paths
    swap = InterestRateSwap(
        notional=1_000_000, fixed_rate=0.04, tenor=T,
        payment_freq=0.25, seed=SEED,
    )
    mtm = swap.simulate_values(paths, times)

    cpty_hazard = hazard_rate_from_cds(0.01, 0.40)
    recovery = 0.40
    lgd = 1.0 - recovery
    positive_exp = np.maximum(mtm, 0.0)

    def compute_wwr_cva(corr: float) -> float:
        """Compute CVA with wrong-way risk adjustment."""
        exp_mean = np.mean(positive_exp, axis=0, keepdims=True)
        exp_std = np.std(positive_exp, axis=0, keepdims=True)
        exp_std = np.where(exp_std < 1e-12, 1.0, exp_std)
        z = (positive_exp - exp_mean) / exp_std
        h_adj = cpty_hazard * np.exp(corr * z)

        cva_val = 0.0
        for i in range(1, len(times)):
            dt_step = times[i] - times[i - 1]
            discount = np.exp(-0.03 * times[i])
            marginal_pd = h_adj[:, i] * dt_step
            cva_val += lgd * float(np.mean(positive_exp[:, i] * marginal_pd)) * discount
        return cva_val

    cva_zero = compute_wwr_cva(0.0)
    cva_positive = compute_wwr_cva(0.5)

    assert cva_positive > cva_zero


# ---------------------------------------------------------------------------
# Test 14: Bilateral CVA = CVA - DVA
# ---------------------------------------------------------------------------

def test_bilateral_cva(profiles):
    """BCVA should equal CVA - DVA."""
    prof, times = profiles
    cpty_hazard = hazard_rate_from_cds(0.01, 0.40)
    own_hazard = hazard_rate_from_cds(0.005, 0.40)
    recovery = 0.40

    result = compute_bilateral_cva(
        ee=prof["ee"],
        ene=prof["ene"],
        times=times,
        cpty_hazard=cpty_hazard,
        own_hazard=own_hazard,
        recovery=recovery,
    )

    assert result["bcva"] == pytest.approx(result["cva"] - result["dva"])
