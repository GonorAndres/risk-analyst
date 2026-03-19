"""Unit tests for Project 12 -- Climate Risk Scenarios.

All tests use synthetic data (no API calls). Covers NGFS pathways,
transition risk, physical risk, Sobol sensitivity, and TCFD reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------
# 1. Synthetic NGFS has 6 pathways
# -----------------------------------------------------------------------

class TestNGFSData:
    """Tests for NGFS scenario data generation."""

    def test_synthetic_has_six_pathways(self, synthetic_scenarios):
        """Synthetic NGFS should contain exactly 6 pathway keys."""
        expected = {
            "net_zero_2050", "below_2c", "low_demand",
            "delayed_transition", "ndcs", "current_policies",
        }
        assert set(synthetic_scenarios.keys()) == expected

    def test_pathway_columns(self, synthetic_scenarios):
        """Each pathway DataFrame should have the required columns."""
        required = {"year", "carbon_price", "temperature_anomaly",
                     "gdp_growth_impact", "sea_level_rise"}
        for name, df in synthetic_scenarios.items():
            assert required.issubset(set(df.columns)), f"Missing columns in {name}"

    def test_pathway_years(self, synthetic_scenarios):
        """Years should span 2025--2100 at 5-year intervals."""
        for name, df in synthetic_scenarios.items():
            years = df["year"].values
            assert years[0] == 2025
            assert years[-1] == 2100
            assert len(years) == 16  # 2025 to 2100 every 5 years


# -----------------------------------------------------------------------
# 2. Carbon cost is non-negative
# -----------------------------------------------------------------------

class TestTransitionRisk:
    """Tests for transition risk computations."""

    def test_carbon_cost_non_negative(self):
        """Carbon cost should be non-negative for non-negative inputs."""
        from transition_risk import compute_carbon_cost
        cost = compute_carbon_cost(intensity=850.0, carbon_price=100.0, revenue=50000.0)
        assert cost >= 0
        assert cost == pytest.approx(850.0 * 100.0 * 50000.0 / 1e6)

    # -------------------------------------------------------------------
    # 3. Transition loss ordering
    # -------------------------------------------------------------------

    def test_transition_loss_ordering(self, sector_data, synthetic_scenarios, portfolio_weights):
        """Current Policies should have lowest transition loss (no carbon price).
        Delayed Transition should have highest (sharp price spike by 2050)."""
        from transition_risk import transition_loss_by_scenario

        df = transition_loss_by_scenario(sector_data, synthetic_scenarios, portfolio_weights)
        losses_2050 = df[df["year"] == 2050].set_index("scenario")["portfolio_loss"]

        # Current Policies has carbon_price ~ 0 -> lowest transition loss
        assert losses_2050["current_policies"] < losses_2050["delayed_transition"]
        assert losses_2050["current_policies"] < losses_2050["net_zero_2050"]

        # Delayed Transition has the highest carbon price by 2050 (~400)
        assert losses_2050["delayed_transition"] == losses_2050.max()

    # -------------------------------------------------------------------
    # 4. WACI is positive and finite
    # -------------------------------------------------------------------

    def test_waci_positive_finite(self, sector_data, portfolio_weights):
        """WACI should be positive and finite."""
        from transition_risk import waci
        result = waci(portfolio_weights, sector_data["carbon_intensity"].values)
        assert result > 0
        assert np.isfinite(result)

    # -------------------------------------------------------------------
    # 7. Stranded fraction between 0 and 1
    # -------------------------------------------------------------------

    def test_stranded_fraction_bounds(self):
        """Stranded asset fraction should be in [0, 1]."""
        from transition_risk import stranded_asset_exposure

        reserves = np.array([100000.0, 50000.0, 30000.0])
        extraction_costs = np.array([35.0, 45.0, 60.0])

        for cp in [0.0, 50.0, 100.0, 200.0, 500.0]:
            fractions = stranded_asset_exposure(
                reserves, extraction_costs, carbon_price=cp, market_price=80.0,
            )
            assert np.all(fractions >= 0.0)
            assert np.all(fractions <= 1.0)

    # -------------------------------------------------------------------
    # 12. Sector repricing produces negative equity impact
    # -------------------------------------------------------------------

    def test_sector_repricing_negative_equity_impact(self, sector_data):
        """With positive carbon price, equity impact should be negative."""
        from transition_risk import sector_repricing
        repriced = sector_repricing(sector_data, carbon_price=100.0)
        assert (repriced["equity_impact"] <= 0).all()
        assert (repriced["carbon_cost"] >= 0).all()


# -----------------------------------------------------------------------
# 5. Temperature damage is convex
# -----------------------------------------------------------------------

class TestPhysicalRisk:
    """Tests for physical risk computations."""

    def test_temperature_damage_convex(self):
        """Damage at 3C should be more than 2x damage at 2C (quadratic)."""
        from physical_risk import temperature_damage_function
        d2 = temperature_damage_function(2.0)
        d3 = temperature_damage_function(3.0)
        # d(3) / d(2) = 9/4 = 2.25 > 2
        assert d3 > 2 * d2

    # -------------------------------------------------------------------
    # 6. Agriculture multiplier > technology multiplier
    # -------------------------------------------------------------------

    def test_sector_multiplier_ordering(self):
        """Agriculture should have higher physical risk than technology."""
        from physical_risk import physical_risk_by_sector
        risks = physical_risk_by_sector(2.0)
        assert risks["agriculture"] > risks["technology"]

    # -------------------------------------------------------------------
    # 13. Physical loss increases with temperature (monotonic)
    # -------------------------------------------------------------------

    def test_physical_loss_monotonic(self, sector_data, synthetic_scenarios, portfolio_weights):
        """Physical loss should increase with temperature anomaly."""
        from physical_risk import physical_loss_by_scenario

        df = physical_loss_by_scenario(sector_data, synthetic_scenarios, portfolio_weights)

        # At 2050: Current Policies (highest temp) should have highest physical loss
        losses_2050 = df[df["year"] == 2050].set_index("scenario")
        temps_2050 = losses_2050["temperature"]
        phys_2050 = losses_2050["portfolio_loss"]

        # Sort by temperature and check loss is non-decreasing
        sorted_idx = temps_2050.sort_values().index
        sorted_losses = phys_2050.loc[sorted_idx].values
        for i in range(len(sorted_losses) - 1):
            assert sorted_losses[i] <= sorted_losses[i + 1] + 1e-8


# -----------------------------------------------------------------------
# 8. Climate VaR is positive
# -----------------------------------------------------------------------

class TestClimateRiskModel:
    """Tests for the orchestrating ClimateRiskModel."""

    def test_climate_var_positive(self, default_config):
        """Climate VaR should be positive."""
        from model import ClimateRiskModel
        m = ClimateRiskModel(config=default_config)
        m.load_data(use_api=False)
        result = m.compute_climate_var(alpha=0.95)
        assert result["var"] > 0
        assert np.isfinite(result["var"])

    # -------------------------------------------------------------------
    # 9. Sobol S1 values are non-negative
    # -------------------------------------------------------------------

    def test_sobol_s1_non_negative(self, default_config):
        """First-order Sobol indices should be non-negative."""
        from model import ClimateRiskModel
        m = ClimateRiskModel(config=default_config)
        m.load_data(use_api=False)
        sobol_df = m.run_sobol()
        assert (sobol_df["S1"] >= 0).all()

    # -------------------------------------------------------------------
    # 10. Sobol ST >= S1 for each factor
    # -------------------------------------------------------------------

    def test_sobol_st_geq_s1(self, default_config):
        """Total-order indices should be >= first-order indices."""
        from model import ClimateRiskModel
        m = ClimateRiskModel(config=default_config)
        m.load_data(use_api=False)
        sobol_df = m.run_sobol()
        for _, row in sobol_df.iterrows():
            assert row["ST"] >= row["S1"] - 1e-6, (
                f"ST < S1 for factor {row['factor']}: ST={row['ST']:.4f}, S1={row['S1']:.4f}"
            )

    # -------------------------------------------------------------------
    # 11. TCFD report is non-empty string
    # -------------------------------------------------------------------

    def test_tcfd_report_non_empty(self, default_config):
        """TCFD report should be a non-empty markdown string."""
        from model import ClimateRiskModel
        m = ClimateRiskModel(config=default_config)
        m.load_data(use_api=False)
        report = m.tcfd_summary()
        assert isinstance(report, str)
        assert len(report) > 100
        assert "TCFD" in report

    # -------------------------------------------------------------------
    # 14. Scenario comparison has correct shape
    # -------------------------------------------------------------------

    def test_scenario_comparison_shape(self, default_config):
        """Scenario comparison should have 6 scenarios x 2 horizons = 12 rows."""
        from model import ClimateRiskModel
        m = ClimateRiskModel(config=default_config)
        m.load_data(use_api=False)
        comparison = m.scenario_comparison()

        # 6 scenarios * 2 horizons (2030, 2050)
        assert len(comparison) == 12
        expected_cols = {"scenario", "year", "transition_loss", "physical_loss", "total_loss"}
        assert expected_cols.issubset(set(comparison.columns))

        # All 6 scenarios present
        assert len(comparison["scenario"].unique()) == 6
