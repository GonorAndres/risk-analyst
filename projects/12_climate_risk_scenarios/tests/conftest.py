"""Shared fixtures for Project 12 -- Climate Risk Scenarios tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path so imports work without installation
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)


@pytest.fixture
def seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(seed)


@pytest.fixture
def synthetic_scenarios(seed: int) -> dict[str, pd.DataFrame]:
    """Six synthetic NGFS scenarios."""
    from ngfs_data import generate_synthetic_ngfs
    return generate_synthetic_ngfs(seed=seed)


@pytest.fixture
def sector_data(seed: int) -> pd.DataFrame:
    """Sector carbon intensity data."""
    from ngfs_data import get_sector_carbon_intensity
    return get_sector_carbon_intensity(seed=seed)


@pytest.fixture
def portfolio_weights() -> np.ndarray:
    """Default portfolio weights for 8 sectors."""
    return np.array([0.10, 0.10, 0.10, 0.15, 0.15, 0.20, 0.10, 0.10])


@pytest.fixture
def default_config() -> dict:
    """Default configuration dict for ClimateRiskModel."""
    return {
        "scenarios": {
            "pathways": [
                "net_zero_2050", "below_2c", "low_demand",
                "delayed_transition", "ndcs", "current_policies",
            ],
            "use_api": False,
            "horizons": [2030, 2050],
        },
        "portfolio": {
            "sectors": [
                "energy", "utilities", "materials", "industrials",
                "financials", "technology", "healthcare", "real_estate",
            ],
            "weights": [0.10, 0.10, 0.10, 0.15, 0.15, 0.20, 0.10, 0.10],
        },
        "transition": {"market_price_carbon_equivalent": 80.0},
        "physical": {"damage_coefficient": 0.00267, "flood_threshold": 0.5},
        "sobol": {
            "n_samples": 512,  # Sufficient for stable Sobol estimates
            "factors": ["carbon_price", "temperature", "gdp_impact", "sea_level_rise"],
            "bounds": {
                "carbon_price": [0, 500],
                "temperature": [1.0, 4.0],
                "gdp_impact": [-0.05, 0.02],
                "sea_level_rise": [0.1, 1.5],
            },
        },
        "risk": {"alpha": 0.95},
        "random_seed": 42,
    }
