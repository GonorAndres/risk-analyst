"""TCFD-aligned climate risk metrics and reporting.

Computes Weighted Average Carbon Intensity (WACI) paths, financed emissions,
and generates markdown scenario analysis reports aligned with TCFD
recommendations.

References:
    - TCFD (2017). Recommendations of the Task Force on Climate-related
      Financial Disclosures.
    - TCFD (2021). Guidance on Metrics, Targets, and Transition Plans.
    - Partnership for Carbon Accounting Financials (PCAF, 2022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Scenario-specific intensity decline assumptions (fraction of base by 2050)
_SCENARIO_INTENSITY_PATHS: dict[str, dict[int, float]] = {
    "net_zero_2050": {2025: 1.0, 2030: 0.70, 2040: 0.35, 2050: 0.20, 2070: 0.10, 2100: 0.05},
    "below_2c": {2025: 1.0, 2030: 0.80, 2040: 0.50, 2050: 0.35, 2070: 0.25, 2100: 0.15},
    "low_demand": {2025: 1.0, 2030: 0.85, 2040: 0.55, 2050: 0.40, 2070: 0.30, 2100: 0.20},
    "delayed_transition": {2025: 1.0, 2030: 0.95, 2040: 0.50, 2050: 0.30, 2070: 0.20, 2100: 0.10},
    "ndcs": {2025: 1.0, 2030: 0.95, 2040: 0.85, 2050: 0.75, 2070: 0.60, 2100: 0.50},
    "current_policies": {2025: 1.0, 2030: 1.02, 2040: 1.05, 2050: 1.10, 2070: 1.15, 2100: 1.20},
}


def compute_waci_path(
    weights: np.ndarray,
    base_intensities: np.ndarray,
    scenario_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute WACI over time for a given scenario.

    Intensities decline (or grow) according to scenario-specific trajectories.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    base_intensities : np.ndarray
        Base carbon intensities (tCO2/$M revenue) at 2025.
    scenario_df : pd.DataFrame
        Scenario DataFrame with a 'year' column. The scenario name is
        inferred from the carbon_price trajectory pattern.

    Returns
    -------
    pd.DataFrame
        Columns: year, waci.
    """
    years = scenario_df["year"].values
    base_waci = float(np.sum(weights * base_intensities))

    # Determine scenario from carbon price pattern
    scenario_name = _infer_scenario_name(scenario_df)
    intensity_path = _SCENARIO_INTENSITY_PATHS.get(scenario_name, {})

    if not intensity_path:
        # Default: no change
        waci_values = np.full(len(years), base_waci)
    else:
        kp_years = sorted(intensity_path.keys())
        kp_fracs = [intensity_path[y] for y in kp_years]
        fractions = np.interp(years, kp_years, kp_fracs)
        waci_values = base_waci * fractions

    return pd.DataFrame({"year": years, "waci": waci_values})


def _infer_scenario_name(scenario_df: pd.DataFrame) -> str:
    """Infer scenario name from carbon price at 2050.

    Simple heuristic based on carbon price levels.
    """
    if "carbon_price" not in scenario_df.columns:
        return "unknown"

    idx_2050 = (scenario_df["year"] - 2050).abs().idxmin()
    cp_2050 = scenario_df.loc[idx_2050, "carbon_price"]

    if cp_2050 > 200:
        idx_2030 = (scenario_df["year"] - 2030).abs().idxmin()
        cp_2030 = scenario_df.loc[idx_2030, "carbon_price"]
        if cp_2030 < 60:
            return "delayed_transition"
        return "net_zero_2050"
    elif cp_2050 > 120:
        return "below_2c"
    elif cp_2050 > 70:
        return "low_demand"
    elif cp_2050 > 20:
        return "ndcs"
    else:
        return "current_policies"


def compute_financed_emissions(
    weights: np.ndarray,
    intensities: np.ndarray,
    portfolio_value: float,
) -> float:
    """Compute total financed emissions (tCO2).

    financed_emissions = sum(w_i * CI_i) * portfolio_value / 1e6

    The portfolio_value is in $M, and intensities are tCO2/$M, so the
    result is in tCO2.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    intensities : np.ndarray
        Carbon intensities (tCO2/$M revenue).
    portfolio_value : float
        Total portfolio value in $M.

    Returns
    -------
    float
        Financed emissions in tCO2.
    """
    waci_val = float(np.sum(weights * intensities))
    return waci_val * portfolio_value / 1e6


def tcfd_report(
    scenario_results: dict,
    portfolio_name: str = "Multi-Sector Portfolio",
) -> str:
    """Generate a TCFD-aligned markdown scenario analysis report.

    Parameters
    ----------
    scenario_results : dict
        Dictionary with keys:
            - "transition_loss": pd.DataFrame from transition_loss_by_scenario
            - "physical_loss": pd.DataFrame from physical_loss_by_scenario
            - "waci": float (current WACI)
            - "financed_emissions": float (current financed emissions)
            - "climate_var": dict with alpha and var value
    portfolio_name : str
        Name of the portfolio for the report header.

    Returns
    -------
    str
        Markdown-formatted TCFD report.
    """
    lines = []
    lines.append(f"# TCFD Scenario Analysis Report: {portfolio_name}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report presents climate-related scenario analysis aligned with "
        "the Task Force on Climate-related Financial Disclosures (TCFD) "
        "recommendations. Six NGFS Phase V pathways are analyzed for both "
        "transition and physical risks."
    )
    lines.append("")

    # Key metrics
    lines.append("## Key Metrics")
    lines.append("")
    waci_val = scenario_results.get("waci", 0.0)
    fe_val = scenario_results.get("financed_emissions", 0.0)
    lines.append(f"- **Weighted Average Carbon Intensity (WACI):** {waci_val:.1f} tCO2/$M revenue")
    lines.append(f"- **Financed Emissions:** {fe_val:,.0f} tCO2")

    climate_var = scenario_results.get("climate_var", {})
    if climate_var:
        alpha = climate_var.get("alpha", 0.95)
        var_val = climate_var.get("var", 0.0)
        lines.append(f"- **Climate VaR ({alpha:.0%}):** {var_val:.4f} (portfolio fraction)")
    lines.append("")

    # Transition risk table
    lines.append("## Transition Risk by Scenario")
    lines.append("")
    transition_df = scenario_results.get("transition_loss", pd.DataFrame())
    if not transition_df.empty:
        lines.append("| Scenario | Year | Portfolio Loss | Worst Sector |")
        lines.append("|----------|------|---------------|-------------|")
        for _, row in transition_df.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['year']:.0f} | "
                f"{row['portfolio_loss']:.4f} | {row['worst_sector']} |"
            )
        lines.append("")

    # Physical risk table
    lines.append("## Physical Risk by Scenario")
    lines.append("")
    physical_df = scenario_results.get("physical_loss", pd.DataFrame())
    if not physical_df.empty:
        lines.append("| Scenario | Year | Portfolio Loss | Temperature (C) |")
        lines.append("|----------|------|---------------|-----------------|")
        for _, row in physical_df.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['year']:.0f} | "
                f"{row['portfolio_loss']:.4f} | {row['temperature']:.2f} |"
            )
        lines.append("")

    # Recommendations
    lines.append("## Risk Management Recommendations")
    lines.append("")
    lines.append(
        "1. **High-emission sectors** (energy, utilities, materials) face "
        "significant transition risk under orderly pathways."
    )
    lines.append(
        "2. **Physical risk** dominates in hot-house scenarios (NDCs, "
        "Current Policies), particularly for real estate and agriculture."
    )
    lines.append(
        "3. **Delayed Transition** poses the highest combined risk due to "
        "a sharp carbon price spike after 2030."
    )
    lines.append(
        "4. Portfolio decarbonization toward **Net Zero 2050** alignment "
        "reduces long-term transition exposure."
    )
    lines.append("")

    return "\n".join(lines)
