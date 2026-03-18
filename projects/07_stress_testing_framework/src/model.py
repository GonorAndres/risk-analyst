"""Stress Test Framework -- top-level orchestrator.

Coordinates DFAST scenario runs, historical crisis replay, reverse stress
testing, and report generation.

References:
    - Federal Reserve, DFAST 2023 Supervisory Scenarios.
    - Basel Committee (2018), Stress Testing Principles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scenarios import get_dfast_scenarios, get_historical_scenarios
from transmission import MacroTransmissionModel
from reverse_stress import reverse_stress_test


class StressTestFramework:
    """End-to-end stress testing orchestrator.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically loaded from ``default.yaml``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.transmission_model = MacroTransmissionModel()
        self._dfast_results: pd.DataFrame | None = None
        self._historical_results: pd.DataFrame | None = None
        self._reverse_result: dict | None = None

    # --------------------------------------------------------------- DFAST
    def run_dfast(
        self,
        portfolio_returns: np.ndarray,
        macro_factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit transmission model and apply all three DFAST scenarios.

        Returns
        -------
        pd.DataFrame
            Columns: scenario, cumulative_loss, max_quarterly_loss,
            capital_ratio_impact.
        """
        self.transmission_model.fit(portfolio_returns, macro_factors)
        scenarios = get_dfast_scenarios()
        initial_ratio = self.config.get("capital", {}).get("initial_ratio", 0.12)

        rows: list[dict] = []
        for name, scenario_df in scenarios.items():
            cum_loss = self.transmission_model.predict_path(scenario_df)
            total_cum = float(cum_loss.iloc[-1])
            max_quarterly = float(cum_loss.diff().fillna(cum_loss.iloc[0]).max())
            capital_impact = initial_ratio - total_cum

            rows.append({
                "scenario": name,
                "cumulative_loss": total_cum,
                "max_quarterly_loss": max_quarterly,
                "capital_ratio_impact": capital_impact,
            })

        self._dfast_results = pd.DataFrame(rows)
        return self._dfast_results

    # ----------------------------------------------------------- Historical
    def run_historical(
        self,
        portfolio_returns: np.ndarray,
        macro_factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply historical crisis scenarios.

        Returns
        -------
        pd.DataFrame
            Columns: scenario, predicted_loss.
        """
        if not self.transmission_model._fitted:
            self.transmission_model.fit(portfolio_returns, macro_factors)

        historical = get_historical_scenarios()
        rows: list[dict] = []
        for name, shocks in historical.items():
            loss = self.transmission_model.predict_loss(shocks)
            rows.append({"scenario": name, "predicted_loss": loss})

        self._historical_results = pd.DataFrame(rows)
        return self._historical_results

    # ----------------------------------------------------- Reverse stress
    def run_reverse(
        self,
        portfolio_returns: np.ndarray,
        macro_factors: pd.DataFrame,
        threshold: float,
    ) -> dict:
        """Run reverse stress test to find break-point scenario.

        Parameters
        ----------
        threshold : float
            Loss threshold to breach (e.g. 0.15 for 15%).

        Returns
        -------
        dict
            Result from :func:`reverse_stress_test`.
        """
        if not self.transmission_model._fitted:
            self.transmission_model.fit(portfolio_returns, macro_factors)

        factor_names = self.config.get("transmission", {}).get(
            "factors",
            list(macro_factors.columns),
        )
        self._reverse_result = reverse_stress_test(
            self.transmission_model,
            loss_threshold=threshold,
            factor_names=factor_names,
        )
        return self._reverse_result

    # -------------------------------------------------------------- Report
    def generate_report(self) -> str:
        """Generate a Markdown stress test report consolidating all results.

        Returns
        -------
        str
            Markdown-formatted report.
        """
        lines: list[str] = [
            "# Stress Test Report",
            "",
            "## 1. Macro Transmission Model",
            "",
        ]

        if self.transmission_model._fitted:
            lines.append(
                f"- R-squared: {self.transmission_model.r_squared:.4f}"
            )
            lines.append(
                f"- Residual std: {self.transmission_model.residual_std:.6f}"
            )
            sens = self.transmission_model.sensitivity_table()
            lines.append("")
            lines.append("### Factor Sensitivities")
            lines.append("")
            lines.append(sens.to_markdown(index=False))
        else:
            lines.append("*Transmission model not yet fitted.*")

        lines.append("")
        lines.append("## 2. DFAST Scenario Results")
        lines.append("")
        if self._dfast_results is not None:
            lines.append(self._dfast_results.to_markdown(index=False))
        else:
            lines.append("*DFAST scenarios not yet executed.*")

        lines.append("")
        lines.append("## 3. Historical Crisis Replay")
        lines.append("")
        if self._historical_results is not None:
            lines.append(self._historical_results.to_markdown(index=False))
        else:
            lines.append("*Historical scenarios not yet executed.*")

        lines.append("")
        lines.append("## 4. Reverse Stress Test")
        lines.append("")
        if self._reverse_result is not None:
            lines.append(f"- Success: {self._reverse_result['success']}")
            lines.append(
                f"- Predicted loss: {self._reverse_result['predicted_loss']:.4f}"
            )
            lines.append(
                f"- Shock norm: {self._reverse_result['shock_norm']:.4f}"
            )
            lines.append("")
            lines.append("### Optimal Shocks")
            lines.append("")
            for k, v in self._reverse_result["optimal_shocks"].items():
                lines.append(f"- {k}: {v:+.4f}")
        else:
            lines.append("*Reverse stress test not yet executed.*")

        lines.append("")
        return "\n".join(lines)
