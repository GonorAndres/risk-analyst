"""Climate Risk Model -- orchestrates NGFS scenario analysis.

Combines transition risk, physical risk, Sobol sensitivity analysis, and
TCFD reporting into a single model class driven by YAML configuration.

References:
    - NGFS (2025). Climate Scenarios -- Phase V.
    - TCFD (2017). Recommendations of the TCFD.
    - Battiston et al. (2017). A climate stress-test of the financial system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ngfs_data import (
    get_sector_carbon_intensity,
    load_ngfs_scenarios,
)
from physical_risk import (
    physical_loss_by_scenario,
    temperature_damage_function,
)
from sobol_analysis import run_sobol_analysis
from tcfd_metrics import (
    compute_financed_emissions,
    compute_waci_path,
    tcfd_report,
)
from transition_risk import (
    transition_loss_by_scenario,
    waci,
)


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class ClimateRiskModel:
    """End-to-end climate risk scenario analysis model.

    Parameters
    ----------
    config : dict
        Model configuration. If None, loads from default.yaml.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            with open(_DEFAULT_CONFIG_PATH) as f:
                config = yaml.safe_load(f)
        self.config = config
        self.seed: int = config.get("random_seed", 42)

        # Portfolio config
        portfolio_cfg = config.get("portfolio", {})
        self.sectors: list[str] = portfolio_cfg.get("sectors", [])
        self.weights: np.ndarray = np.array(
            portfolio_cfg.get("weights", []),
            dtype=np.float64,
        )

        # Data placeholders
        self.scenarios: dict[str, pd.DataFrame] = {}
        self.sector_data: pd.DataFrame = pd.DataFrame()
        self._data_loaded: bool = False

    def load_data(self, use_api: bool = False) -> None:
        """Load NGFS scenarios and sector carbon intensity data.

        Parameters
        ----------
        use_api : bool
            If True, attempt IIASA API download (falls back to synthetic).
        """
        result = load_ngfs_scenarios(
            use_api=use_api,
            seed=self.seed,
        )
        self.scenarios = result["scenarios"]
        self.sector_data = get_sector_carbon_intensity(seed=self.seed)
        self._data_loaded = True

    def _ensure_data(self) -> None:
        """Load data if not already loaded."""
        if not self._data_loaded:
            self.load_data(use_api=False)

    def compute_transition_risk(self) -> pd.DataFrame:
        """Compute transition losses by scenario at 2030 and 2050.

        Returns
        -------
        pd.DataFrame
            Columns: scenario, year, portfolio_loss, worst_sector.
        """
        self._ensure_data()
        return transition_loss_by_scenario(
            self.sector_data,
            self.scenarios,
            self.weights,
        )

    def compute_physical_risk(self) -> pd.DataFrame:
        """Compute physical losses by scenario at 2030 and 2050.

        Returns
        -------
        pd.DataFrame
            Columns: scenario, year, portfolio_loss, temperature, sea_level_rise.
        """
        self._ensure_data()
        return physical_loss_by_scenario(
            self.sector_data,
            self.scenarios,
            self.weights,
        )

    def compute_climate_var(self, alpha: float | None = None) -> dict:
        """Compute Climate VaR combining transition and physical risks.

        Aggregates losses across all scenarios and horizons, then takes
        the (1-alpha) quantile as the Climate VaR.

        Parameters
        ----------
        alpha : float, optional
            Confidence level (default from config, typically 0.95).

        Returns
        -------
        dict
            Keys: alpha, var, es (expected shortfall), all_losses.
        """
        if alpha is None:
            alpha = self.config.get("risk", {}).get("alpha", 0.95)

        transition_df = self.compute_transition_risk()
        physical_df = self.compute_physical_risk()

        # Combine losses: for each (scenario, year) pair, sum transition + physical
        combined = transition_df[["scenario", "year", "portfolio_loss"]].copy()
        combined = combined.rename(columns={"portfolio_loss": "transition_loss"})

        physical_merged = physical_df[["scenario", "year", "portfolio_loss"]].copy()
        physical_merged = physical_merged.rename(columns={"portfolio_loss": "physical_loss"})

        merged = combined.merge(physical_merged, on=["scenario", "year"])
        merged["total_loss"] = merged["transition_loss"] + merged["physical_loss"]

        all_losses = merged["total_loss"].values
        var_val = float(np.quantile(all_losses, alpha))
        es_val = float(np.mean(all_losses[all_losses >= var_val])) if np.any(all_losses >= var_val) else var_val

        return {
            "alpha": alpha,
            "var": var_val,
            "es": es_val,
            "all_losses": all_losses,
        }

    def run_sobol(self) -> pd.DataFrame:
        """Run Sobol sensitivity analysis on climate factors.

        The model function maps (carbon_price, temperature, gdp_impact,
        sea_level_rise) to a combined portfolio loss.

        Returns
        -------
        pd.DataFrame
            Columns: factor, S1, ST.
        """
        self._ensure_data()
        sobol_cfg = self.config.get("sobol", {})
        n_samples = sobol_cfg.get("n_samples", 512)
        factors = sobol_cfg.get("factors", [])
        bounds_cfg = sobol_cfg.get("bounds", {})
        bounds = [tuple(bounds_cfg[f]) for f in factors]

        sector_data = self.sector_data
        weights = self.weights
        damage_coeff = self.config.get("physical", {}).get("damage_coefficient", 0.00267)

        def climate_loss_model(x: np.ndarray) -> float:
            """Map climate factors to portfolio loss."""
            carbon_price = x[0]
            temperature = x[1]
            gdp_impact = x[2]
            sea_level_rise = x[3]

            # Transition loss component
            intensities = sector_data["carbon_intensity"].values
            ebitda_margins = sector_data["ebitda_margin"].values
            revenues = sector_data["revenue"].values
            carbon_costs = intensities * carbon_price * revenues / 1e6
            equity_impacts = carbon_costs / (ebitda_margins * revenues)
            transition_loss = float(np.sum(weights * equity_impacts))

            # Physical loss component
            physical_damage = damage_coeff * temperature ** 2
            multipliers = np.array([
                0.8, 1.2, 1.0, 0.9, 0.5, 0.2, 0.3, 1.5,
            ])
            physical_loss = float(np.sum(weights * multipliers * physical_damage))

            # GDP drag
            gdp_loss = abs(gdp_impact) * 0.5

            # Sea level rise component
            slr_loss = max(0.0, (sea_level_rise - 0.5) / 0.5) * 0.02

            return transition_loss + physical_loss + gdp_loss + slr_loss

        return run_sobol_analysis(
            model_fn=climate_loss_model,
            factor_names=factors,
            bounds=bounds,
            n_samples=n_samples,
            seed=self.seed,
        )

    def scenario_comparison(self) -> pd.DataFrame:
        """Compare all scenarios side by side.

        Returns
        -------
        pd.DataFrame
            Columns: scenario, year, transition_loss, physical_loss, total_loss.
        """
        transition_df = self.compute_transition_risk()
        physical_df = self.compute_physical_risk()

        combined = transition_df[["scenario", "year", "portfolio_loss"]].copy()
        combined = combined.rename(columns={"portfolio_loss": "transition_loss"})

        physical_sub = physical_df[["scenario", "year", "portfolio_loss"]].copy()
        physical_sub = physical_sub.rename(columns={"portfolio_loss": "physical_loss"})

        merged = combined.merge(physical_sub, on=["scenario", "year"])
        merged["total_loss"] = merged["transition_loss"] + merged["physical_loss"]

        return merged

    def tcfd_summary(self) -> str:
        """Generate TCFD markdown report.

        Returns
        -------
        str
            Markdown-formatted TCFD scenario analysis report.
        """
        self._ensure_data()
        transition_df = self.compute_transition_risk()
        physical_df = self.compute_physical_risk()
        climate_var = self.compute_climate_var()

        waci_val = waci(self.weights, self.sector_data["carbon_intensity"].values)
        fe_val = compute_financed_emissions(
            self.weights,
            self.sector_data["carbon_intensity"].values,
            portfolio_value=1000.0,  # $1B portfolio
        )

        return tcfd_report(
            scenario_results={
                "transition_loss": transition_df,
                "physical_loss": physical_df,
                "waci": waci_val,
                "financed_emissions": fe_val,
                "climate_var": climate_var,
            },
            portfolio_name="Multi-Sector Portfolio",
        )
