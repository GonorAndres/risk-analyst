"""Transition risk quantification under climate scenarios.

Models the financial impact of carbon pricing on sector-level equity through
carbon cost attribution, stranded asset exposure, and weighted average carbon
intensity (WACI).

References:
    - Battiston et al. (2017). A climate stress-test of the financial system.
    - TCFD (2017). Recommendations of the Task Force on Climate-related
      Financial Disclosures.
    - Bolton & Kacperczyk (2021). Do investors care about carbon risk?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_carbon_cost(
    intensity: float,
    carbon_price: float,
    revenue: float,
) -> float:
    """Compute annual carbon cost for a sector.

    cost = intensity * carbon_price * revenue / 1e6

    Parameters
    ----------
    intensity : float
        Carbon intensity in tCO2 per $M revenue.
    carbon_price : float
        Carbon price in USD per tCO2.
    revenue : float
        Sector revenue in $M.

    Returns
    -------
    float
        Annual carbon cost in $M.
    """
    return intensity * carbon_price * revenue / 1e6


def sector_repricing(
    sector_data: pd.DataFrame,
    carbon_price: float,
) -> pd.DataFrame:
    """Reprice sectors under a given carbon price.

    Adds columns:
        carbon_cost -- annual carbon cost ($M)
        equity_impact -- fractional equity impact (negative = loss)

    Parameters
    ----------
    sector_data : pd.DataFrame
        Indexed by sector with columns: carbon_intensity, ebitda_margin,
        revenue.
    carbon_price : float
        Carbon price in USD per tCO2.

    Returns
    -------
    pd.DataFrame
        Copy of sector_data with carbon_cost and equity_impact columns.
    """
    df = sector_data.copy()
    df["carbon_cost"] = df.apply(
        lambda row: compute_carbon_cost(
            row["carbon_intensity"], carbon_price, row["revenue"],
        ),
        axis=1,
    )
    # Equity impact: loss as fraction of EBITDA-implied equity value
    df["equity_impact"] = -df["carbon_cost"] / (
        df["ebitda_margin"] * df["revenue"]
    )
    return df


def stranded_asset_exposure(
    reserves: np.ndarray,
    extraction_cost: np.ndarray,
    carbon_price: float,
    market_price: float = 80.0,
) -> np.ndarray:
    """Fraction of reserves stranded under a carbon price.

    An asset is stranded when extraction_cost + carbon_price > market_price,
    making extraction uneconomic.

    Parameters
    ----------
    reserves : np.ndarray
        Reserve values (not used in fraction calculation, but represents
        the assets at risk).
    extraction_cost : np.ndarray
        Extraction cost per unit ($/tCO2 equivalent) for each asset.
    carbon_price : float
        Carbon price in USD per tCO2.
    market_price : float
        Market price of the commodity in $/unit.

    Returns
    -------
    np.ndarray
        Fraction stranded for each asset, clipped to [0, 1].
    """
    total_cost = extraction_cost + carbon_price
    # Fraction stranded: how far above market price the total cost is,
    # relative to market price
    fraction = np.where(
        market_price > 0,
        np.clip((total_cost - market_price) / market_price, 0.0, 1.0),
        np.where(total_cost > 0, 1.0, 0.0),
    )
    return fraction.astype(np.float64)


def waci(
    weights: np.ndarray,
    intensities: np.ndarray,
) -> float:
    """Weighted Average Carbon Intensity (WACI).

    WACI = sum(w_i * CI_i)

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (should sum to 1).
    intensities : np.ndarray
        Carbon intensities (tCO2/$M revenue) per holding.

    Returns
    -------
    float
        Portfolio-level WACI.
    """
    return float(np.sum(weights * intensities))


def transition_loss_by_scenario(
    sector_data: pd.DataFrame,
    scenarios: dict[str, pd.DataFrame],
    weights: np.ndarray,
) -> pd.DataFrame:
    """Compute portfolio-weighted transition loss for each scenario at 2030 and 2050.

    Parameters
    ----------
    sector_data : pd.DataFrame
        Indexed by sector with carbon_intensity, ebitda_margin, revenue.
    scenarios : dict[str, pd.DataFrame]
        Mapping of scenario names to DataFrames with year and carbon_price.
    weights : np.ndarray
        Portfolio weights aligned with sector_data index.

    Returns
    -------
    pd.DataFrame
        Columns: scenario, year, portfolio_loss, worst_sector.
    """
    results = []
    for scenario_name, scenario_df in scenarios.items():
        for horizon in [2030, 2050]:
            # Get carbon price at horizon (nearest year)
            idx = (scenario_df["year"] - horizon).abs().idxmin()
            carbon_price = scenario_df.loc[idx, "carbon_price"]

            # Reprice sectors
            repriced = sector_repricing(sector_data, carbon_price)

            # Portfolio-weighted loss (equity_impact is negative, so loss is positive)
            sector_losses = -repriced["equity_impact"].values
            portfolio_loss = float(np.sum(weights * sector_losses))

            # Worst sector
            worst_idx = np.argmax(sector_losses)
            worst_sector = repriced.index[worst_idx]

            results.append({
                "scenario": scenario_name,
                "year": horizon,
                "portfolio_loss": portfolio_loss,
                "worst_sector": worst_sector,
            })

    return pd.DataFrame(results)
