"""Portfolio Risk Dashboard -- main orchestration module.

Loads configuration, fetches market data, computes all risk measures,
runs backtesting, and produces visualizations.

Usage
-----
    from projects.p01_portfolio_risk_dashboard.src.dashboard import PortfolioRiskDashboard

    dashboard = PortfolioRiskDashboard("configs/default.yaml")
    dashboard.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from risk_analyst.data.market import compute_returns, fetch_prices
from risk_analyst.measures.backtesting import BacktestReport, backtest_var
from risk_analyst.utils.config import load_yaml

try:
    from .model import RiskModel
except ImportError:
    from model import RiskModel  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Pydantic configuration schema
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Data ingestion parameters."""

    tickers: list[str]
    start_date: str
    end_date: str | None = None
    frequency: str = "1d"


class RiskMeasuresConfig(BaseModel):
    """Risk measure computation parameters."""

    confidence_levels: list[float] = Field(default=[0.95, 0.99])
    var_methods: list[Literal["historical", "parametric", "monte_carlo"]] = Field(
        default=["historical", "parametric", "monte_carlo"]
    )
    monte_carlo_simulations: int = 10_000
    rolling_window: int = 252
    ewma_lambda: float = 0.94


class BacktestingConfig(BaseModel):
    """Backtesting parameters."""

    test_window: int = 250
    kupiec_significance: float = 0.05
    christoffersen_significance: float = 0.05


class VisualizationConfig(BaseModel):
    """Visualization parameters."""

    theme: str = "plotly_white"
    rolling_vol_window: int = 21


class DashboardConfig(BaseModel):
    """Top-level configuration for the Portfolio Risk Dashboard."""

    data: DataConfig
    risk_measures: RiskMeasuresConfig = RiskMeasuresConfig()
    backtesting: BacktestingConfig = BacktestingConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class RiskMeasureResult:
    """Result of a single VaR/ES computation."""

    method: str
    alpha: float
    var: float
    es: float


@dataclass
class DashboardResults:
    """Aggregated results from a full dashboard run."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    losses: pd.Series
    risk_measures: list[RiskMeasureResult] = field(default_factory=list)
    backtest_reports: dict[str, BacktestReport] = field(default_factory=dict)
    rolling_var: dict[str, pd.Series] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------


class PortfolioRiskDashboard:
    """Main orchestration class for the Portfolio Risk Dashboard.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.
    weights : list[float] | None
        Portfolio weights.  If ``None``, equal weights are used.
    """

    def __init__(
        self,
        config_path: str | Path,
        weights: list[float] | None = None,
    ) -> None:
        raw = load_yaml(config_path)
        self.config = DashboardConfig(**raw)

        n_assets = len(self.config.data.tickers)
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float64)
        else:
            self.weights = np.ones(n_assets) / n_assets

        self.model = RiskModel(
            n_sims=self.config.risk_measures.monte_carlo_simulations,
            seed=self.config.random_seed,
        )
        self.results: DashboardResults | None = None

    def run(self) -> DashboardResults:
        """Execute the full dashboard pipeline.

        1. Fetch prices
        2. Compute returns and losses
        3. Fit risk model
        4. Compute VaR/ES for all method x alpha combinations
        5. Run rolling VaR and backtesting
        6. Store results

        Returns
        -------
        DashboardResults
            All computed results.
        """
        cfg = self.config

        # 1. Fetch prices
        prices = fetch_prices(
            tickers=cfg.data.tickers,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )

        # 2. Compute returns and fit model
        returns = compute_returns(prices, method="log")
        self.model.fit(returns, self.weights)
        losses = self.model.loss_series

        # 3. Initialize results
        self.results = DashboardResults(
            prices=prices,
            returns=returns,
            losses=losses,
        )

        # 4. Compute risk measures
        self.results.risk_measures = self.compute_risk_measures()

        # 5. Rolling VaR and backtesting
        self.results.rolling_var, self.results.backtest_reports = self.run_backtest()

        return self.results

    def compute_risk_measures(self) -> list[RiskMeasureResult]:
        """Compute VaR and ES for all configured methods and alpha levels.

        Returns
        -------
        list[RiskMeasureResult]
            One result per (method, alpha) combination.
        """
        cfg = self.config.risk_measures
        results: list[RiskMeasureResult] = []

        for alpha in cfg.confidence_levels:
            for method in cfg.var_methods:
                var_val = self.model.var(alpha=alpha, method=method)
                es_val = self.model.es(alpha=alpha)
                results.append(
                    RiskMeasureResult(method=method, alpha=alpha, var=var_val, es=es_val)
                )

        return results

    def run_backtest(
        self,
    ) -> tuple[dict[str, pd.Series], dict[str, BacktestReport]]:
        """Run rolling VaR and backtesting for all configured methods.

        Returns
        -------
        tuple[dict[str, pd.Series], dict[str, BacktestReport]]
            Rolling VaR series and backtest reports, keyed by
            "{method}_{alpha}".
        """
        cfg = self.config
        rolling_vars: dict[str, pd.Series] = {}
        reports: dict[str, BacktestReport] = {}

        for alpha in cfg.risk_measures.confidence_levels:
            for method in cfg.risk_measures.var_methods:
                key = f"{method}_{alpha}"
                var_s = self.model.rolling_var(
                    window=cfg.risk_measures.rolling_window,
                    alpha=alpha,
                    method=method,
                )
                rolling_vars[key] = var_s

                # Backtest only on the non-NaN portion
                valid = ~var_s.isna()
                if valid.sum() > 0:
                    losses_valid = self.model.losses[valid.values]
                    var_valid = var_s[valid].values
                    report = backtest_var(
                        losses=losses_valid,
                        var_series=var_valid,
                        alpha=alpha,
                        kupiec_significance=cfg.backtesting.kupiec_significance,
                        christoffersen_significance=cfg.backtesting.christoffersen_significance,
                    )
                    reports[key] = report

        return rolling_vars, reports

    def summary_table(self) -> pd.DataFrame:
        """Build a summary DataFrame of all risk measures.

        Returns
        -------
        pd.DataFrame
            Columns: method, alpha, VaR, ES.
        """
        if self.results is None:
            raise RuntimeError("Dashboard has not been run yet. Call .run() first.")

        rows = []
        for rm in self.results.risk_measures:
            rows.append(
                {
                    "method": rm.method,
                    "alpha": rm.alpha,
                    "VaR": rm.var,
                    "ES": rm.es,
                }
            )
        return pd.DataFrame(rows)

    def backtest_summary_table(self) -> pd.DataFrame:
        """Build a summary DataFrame of backtesting results.

        Returns
        -------
        pd.DataFrame
            One row per (method, alpha) with violation stats and test results.
        """
        if self.results is None:
            raise RuntimeError("Dashboard has not been run yet. Call .run() first.")

        rows = []
        for key, report in self.results.backtest_reports.items():
            rows.append(
                {
                    "key": key,
                    "n_violations": report.n_violations,
                    "n_obs": report.n_obs,
                    "violation_rate": report.violation_rate,
                    "expected_rate": report.expected_rate,
                    "kupiec_LR": report.kupiec.statistic,
                    "kupiec_p": report.kupiec.p_value,
                    "kupiec_reject": report.kupiec.reject,
                    "christoffersen_LR": report.christoffersen.statistic,
                    "christoffersen_p": report.christoffersen.p_value,
                    "christoffersen_reject": report.christoffersen.reject,
                    "traffic_light": report.traffic_light.zone,
                }
            )
        return pd.DataFrame(rows)
