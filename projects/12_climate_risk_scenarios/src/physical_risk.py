"""Physical risk modeling under climate scenarios.

Implements temperature-based damage functions (Nordhaus quadratic), flood
damage models from sea-level rise, and sector-level physical risk multipliers.

References:
    - Nordhaus (2017). Revisiting the social cost of carbon.
    - Burke, Hsiang & Miguel (2015). Global non-linear effect of
      temperature on economic production.
    - Dietz & Stern (2015). Endogenous growth, convexity of damage
      and climate risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temperature_damage_function(
    temp_anomaly: float,
    damage_coeff: float = 0.00267,
) -> float:
    """Nordhaus quadratic damage function.

    fraction_loss = damage_coeff * temp_anomaly^2

    Parameters
    ----------
    temp_anomaly : float
        Temperature anomaly in Celsius above pre-industrial levels.
    damage_coeff : float
        Damage coefficient (Nordhaus DICE default: ~0.00267).

    Returns
    -------
    float
        Fraction of GDP lost to climate damage.
    """
    return damage_coeff * temp_anomaly ** 2


def flood_damage(
    sea_level_rise: float,
    exposure: float,
    threshold: float = 0.5,
) -> float:
    """Flood damage from sea-level rise exceeding a threshold.

    damage = exposure * max(0, (slr - threshold) / threshold)

    Parameters
    ----------
    sea_level_rise : float
        Projected sea-level rise in meters.
    exposure : float
        Monetary exposure to flooding ($M or fraction).
    threshold : float
        Sea-level rise threshold above which damage begins (meters).

    Returns
    -------
    float
        Flood damage value.
    """
    excess = max(0.0, (sea_level_rise - threshold) / threshold)
    return exposure * excess


def physical_risk_by_sector(temp_anomaly: float) -> dict[str, float]:
    """Sector-level physical risk multipliers scaled by temperature.

    Returns the fractional damage for each sector, computed as:
        sector_damage = base_multiplier * damage_function(temp_anomaly)

    Parameters
    ----------
    temp_anomaly : float
        Temperature anomaly (C above pre-industrial).

    Returns
    -------
    dict[str, float]
        Mapping of sector -> fractional physical risk damage.
    """
    base_damage = temperature_damage_function(temp_anomaly)

    # Sector sensitivity multipliers (relative to aggregate)
    multipliers = {
        "agriculture": 2.5,
        "insurance": 2.0,
        "real_estate": 1.5,
        "utilities": 1.2,
        "energy": 0.8,
        "financials": 0.5,
        "technology": 0.2,
        "healthcare": 0.3,
    }

    return {sector: mult * base_damage for sector, mult in multipliers.items()}


def physical_loss_by_scenario(
    sector_data: pd.DataFrame,
    scenarios: dict[str, pd.DataFrame],
    weights: np.ndarray,
) -> pd.DataFrame:
    """Compute portfolio-weighted physical loss for each scenario at 2030 and 2050.

    Uses the Nordhaus damage function scaled by sector multipliers.

    Parameters
    ----------
    sector_data : pd.DataFrame
        Indexed by sector with revenue column.
    scenarios : dict[str, pd.DataFrame]
        Mapping of scenario names to DataFrames with year,
        temperature_anomaly, sea_level_rise.
    weights : np.ndarray
        Portfolio weights aligned with sector_data index.

    Returns
    -------
    pd.DataFrame
        Columns: scenario, year, portfolio_loss, temperature, sea_level_rise.
    """
    # Sector multipliers (subset for portfolio sectors)
    all_multipliers = {
        "energy": 0.8,
        "utilities": 1.2,
        "materials": 1.0,
        "industrials": 0.9,
        "financials": 0.5,
        "technology": 0.2,
        "healthcare": 0.3,
        "real_estate": 1.5,
    }

    results = []
    for scenario_name, scenario_df in scenarios.items():
        for horizon in [2030, 2050]:
            idx = (scenario_df["year"] - horizon).abs().idxmin()
            temp = scenario_df.loc[idx, "temperature_anomaly"]
            slr = scenario_df.loc[idx, "sea_level_rise"]

            base_damage = temperature_damage_function(temp)

            # Sector-level physical losses
            sector_losses = np.zeros(len(sector_data))
            for i, sector in enumerate(sector_data.index):
                mult = all_multipliers.get(sector, 1.0)
                # Combine temperature damage + flood for real_estate
                sector_damage = mult * base_damage
                if sector == "real_estate":
                    sector_damage += flood_damage(slr, 0.1)
                sector_losses[i] = sector_damage

            portfolio_loss = float(np.sum(weights * sector_losses))

            results.append({
                "scenario": scenario_name,
                "year": horizon,
                "portfolio_loss": portfolio_loss,
                "temperature": temp,
                "sea_level_rise": slr,
            })

    return pd.DataFrame(results)
