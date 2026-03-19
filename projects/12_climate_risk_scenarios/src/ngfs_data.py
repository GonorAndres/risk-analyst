"""NGFS climate scenario data loading and synthetic generation.

Provides six NGFS Phase V pathways (Net Zero 2050, Below 2C, Low Demand,
Delayed Transition, NDCs, Current Policies) with macro-financial variables
projected to 2100.  Also includes sector-level carbon intensity data for
transition risk analysis.

References:
    - NGFS (2025). Climate Scenarios -- Phase V.
    - IIASA NGFS Scenario Explorer: data.ece.iiasa.ac.at
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpolate_path(
    years: np.ndarray,
    keypoints: dict[int, float],
) -> np.ndarray:
    """Linearly interpolate between year-keyed control points."""
    kp_years = sorted(keypoints.keys())
    kp_vals = [keypoints[y] for y in kp_years]
    return np.interp(years, kp_years, kp_vals)


# ---------------------------------------------------------------------------
# 1. Synthetic NGFS pathways
# ---------------------------------------------------------------------------

def generate_synthetic_ngfs(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate six synthetic NGFS Phase V pathways with realistic values.

    Each pathway DataFrame has columns:
        year, carbon_price, temperature_anomaly, gdp_growth_impact,
        sea_level_rise.

    Years span 2025--2100 at 5-year intervals (16 points).

    Parameters
    ----------
    seed : int
        Random seed for small jitter added to smooth the curves.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are the six NGFS pathway identifiers.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2025, 2101, 5)

    # Define control points for each pathway:
    #   carbon_price (USD/tCO2), temperature_anomaly (C above pre-industrial),
    #   gdp_growth_impact (%), sea_level_rise (m)

    pathway_specs: dict[str, dict[str, dict[int, float]]] = {
        "net_zero_2050": {
            "carbon_price": {2025: 50, 2030: 100, 2040: 200, 2050: 250, 2070: 300, 2100: 350},
            "temperature_anomaly": {2025: 1.1, 2030: 1.2, 2050: 1.4, 2070: 1.45, 2100: 1.5},
            "gdp_growth_impact": {2025: -0.002, 2030: -0.005, 2050: -0.003, 2070: -0.001, 2100: 0.0},
            "sea_level_rise": {2025: 0.10, 2050: 0.20, 2100: 0.35},
        },
        "below_2c": {
            "carbon_price": {2025: 30, 2030: 60, 2040: 120, 2050: 150, 2070: 180, 2100: 200},
            "temperature_anomaly": {2025: 1.1, 2030: 1.2, 2050: 1.5, 2070: 1.6, 2100: 1.7},
            "gdp_growth_impact": {2025: -0.001, 2030: -0.003, 2050: -0.005, 2070: -0.004, 2100: -0.003},
            "sea_level_rise": {2025: 0.10, 2050: 0.22, 2100: 0.40},
        },
        "low_demand": {
            "carbon_price": {2025: 20, 2030: 40, 2040: 70, 2050: 100, 2070: 120, 2100: 130},
            "temperature_anomaly": {2025: 1.0, 2030: 1.0, 2050: 1.05, 2070: 1.08, 2100: 1.1},
            "gdp_growth_impact": {2025: 0.001, 2030: 0.003, 2050: 0.005, 2070: 0.004, 2100: 0.003},
            "sea_level_rise": {2025: 0.08, 2050: 0.15, 2100: 0.25},
        },
        "delayed_transition": {
            "carbon_price": {2025: 25, 2030: 50, 2035: 200, 2040: 350, 2050: 400, 2070: 380, 2100: 350},
            "temperature_anomaly": {2025: 1.1, 2030: 1.3, 2050: 1.6, 2070: 1.7, 2100: 1.8},
            "gdp_growth_impact": {2025: 0.0, 2030: -0.005, 2035: -0.025, 2040: -0.020, 2050: -0.010, 2070: -0.005, 2100: -0.003},
            "sea_level_rise": {2025: 0.10, 2050: 0.25, 2100: 0.45},
        },
        "ndcs": {
            "carbon_price": {2025: 15, 2030: 30, 2050: 50, 2070: 55, 2100: 60},
            "temperature_anomaly": {2025: 1.1, 2030: 1.3, 2050: 1.8, 2070: 2.2, 2100: 2.6},
            "gdp_growth_impact": {2025: 0.0, 2030: -0.002, 2050: -0.008, 2070: -0.015, 2100: -0.020},
            "sea_level_rise": {2025: 0.10, 2050: 0.30, 2100: 0.65},
        },
        "current_policies": {
            "carbon_price": {2025: 0, 2030: 0, 2050: 0, 2070: 0, 2100: 0},
            "temperature_anomaly": {2025: 1.1, 2030: 1.3, 2050: 2.0, 2070: 2.6, 2100: 3.2},
            "gdp_growth_impact": {2025: 0.0, 2030: -0.003, 2050: -0.015, 2070: -0.030, 2100: -0.045},
            "sea_level_rise": {2025: 0.10, 2050: 0.35, 2100: 0.90},
        },
    }

    scenarios: dict[str, pd.DataFrame] = {}
    for name, spec in pathway_specs.items():
        df = pd.DataFrame({"year": years})
        for col, keypoints in spec.items():
            base = _interpolate_path(years, keypoints)
            # Add small jitter for realism (< 1% of range)
            jitter = rng.normal(0, np.ptp(base) * 0.005, size=len(years))
            df[col] = base + jitter
        # Enforce non-negative carbon price
        df["carbon_price"] = df["carbon_price"].clip(lower=0.0)
        scenarios[name] = df

    return scenarios


# ---------------------------------------------------------------------------
# 2. Sector carbon intensity data
# ---------------------------------------------------------------------------

def get_sector_carbon_intensity(seed: int = 42) -> pd.DataFrame:
    """Return sector-level carbon intensity and financial data.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Indexed by sector with columns: carbon_intensity (tCO2/$M revenue),
        ebitda_margin, revenue ($M), reserves_value ($M), extraction_cost
        ($/tCO2 equivalent).
    """
    rng = np.random.default_rng(seed)

    sectors = [
        "energy", "utilities", "materials", "industrials",
        "financials", "technology", "healthcare", "real_estate",
    ]

    # Approximate carbon intensities (tCO2 per $M revenue)
    carbon_intensities = {
        "energy": 850.0,
        "utilities": 620.0,
        "materials": 450.0,
        "industrials": 180.0,
        "financials": 15.0,
        "technology": 25.0,
        "healthcare": 35.0,
        "real_estate": 120.0,
    }

    ebitda_margins = {
        "energy": 0.25,
        "utilities": 0.20,
        "materials": 0.18,
        "industrials": 0.15,
        "financials": 0.30,
        "technology": 0.28,
        "healthcare": 0.22,
        "real_estate": 0.35,
    }

    revenues = {
        "energy": 50000.0,
        "utilities": 30000.0,
        "materials": 25000.0,
        "industrials": 40000.0,
        "financials": 60000.0,
        "technology": 80000.0,
        "healthcare": 35000.0,
        "real_estate": 20000.0,
    }

    # Reserves and extraction cost only for energy and mining-like sectors
    reserves_values = {
        "energy": 120000.0,
        "utilities": 15000.0,
        "materials": 30000.0,
        "industrials": 0.0,
        "financials": 0.0,
        "technology": 0.0,
        "healthcare": 0.0,
        "real_estate": 0.0,
    }

    extraction_costs = {
        "energy": 35.0,
        "utilities": 45.0,
        "materials": 40.0,
        "industrials": 0.0,
        "financials": 0.0,
        "technology": 0.0,
        "healthcare": 0.0,
        "real_estate": 0.0,
    }

    # Add small jitter
    data = []
    for s in sectors:
        data.append({
            "sector": s,
            "carbon_intensity": carbon_intensities[s] * (1 + rng.normal(0, 0.02)),
            "ebitda_margin": ebitda_margins[s],
            "revenue": revenues[s],
            "reserves_value": reserves_values[s],
            "extraction_cost": extraction_costs[s],
        })

    df = pd.DataFrame(data).set_index("sector")
    return df


# ---------------------------------------------------------------------------
# 3. API download attempt
# ---------------------------------------------------------------------------

def download_ngfs_scenarios() -> dict:
    """Attempt to download NGFS scenarios from the IIASA database.

    Raises
    ------
    NotImplementedError
        Always raised with a helpful message about accessing
        data.ece.iiasa.ac.at for NGFS Phase V data.
    """
    raise NotImplementedError(
        "Automated NGFS download is not implemented. "
        "To access NGFS Phase V scenario data, visit "
        "https://data.ece.iiasa.ac.at and request access "
        "to the NGFS scenario database. Download the CSV "
        "and place it in the project data/ directory."
    )


# ---------------------------------------------------------------------------
# 4. Dispatcher
# ---------------------------------------------------------------------------

def load_ngfs_scenarios(
    use_api: bool = False,
    seed: int = 42,
) -> dict:
    """Load NGFS scenarios: API download or synthetic fallback.

    Parameters
    ----------
    use_api : bool
        If True, attempt to download from IIASA. Falls back to synthetic
        on failure.
    seed : int
        Random seed for synthetic data generation.

    Returns
    -------
    dict
        Contains key ``"scenarios"`` mapping pathway names to DataFrames,
        and ``"source"`` indicating the data origin.
    """
    if use_api:
        try:
            data = download_ngfs_scenarios()
            return {"scenarios": data, "source": "iiasa_api"}
        except NotImplementedError:
            pass

    scenarios = generate_synthetic_ngfs(seed=seed)
    return {"scenarios": scenarios, "source": "synthetic"}
