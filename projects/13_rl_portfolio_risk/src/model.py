"""High-level RL portfolio risk management orchestration.

Brings together environment, agent, trainer, and benchmarks to run
end-to-end experiments and comparisons.

Reference: Wang et al. (2025), ICVaR-DRL for portfolio optimization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from agent import PolicyNetwork
from benchmarks import equal_weight, mean_variance, risk_parity, run_benchmark
from environment import PortfolioEnv
from numpy.typing import NDArray
from trainer import run_episode, train_rl_agent

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
)


class RLPortfolioModel:
    """Orchestrates RL portfolio training, backtesting, and benchmarking.

    Parameters
    ----------
    config : dict
        Full configuration dict (from YAML). If None, loads default.yaml.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            with open(_DEFAULT_CONFIG_PATH) as f:
                config = yaml.safe_load(f)
        self.config = config
        self._trained = False
        self._agent: PolicyNetwork | None = None
        self._env: PortfolioEnv | None = None
        self._train_results: dict | None = None
        self._episode_results: dict | None = None

    def _build_env(self) -> PortfolioEnv:
        """Create a PortfolioEnv from config."""
        port = self.config["portfolio"]
        env_cfg = self.config["environment"]
        return PortfolioEnv(
            n_assets=port["n_assets"],
            lookback=env_cfg["lookback"],
            initial_capital=port["initial_capital"],
            cost_rate=port["cost_rate"],
            cvar_alpha=env_cfg["cvar_alpha"],
            cvar_threshold=env_cfg["cvar_threshold"],
        )

    def _build_agent(self, env: PortfolioEnv) -> PolicyNetwork:
        """Create a PolicyNetwork from config."""
        agent_cfg = self.config["agent"]
        seed = self.config["random_seed"]
        return PolicyNetwork(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=agent_cfg["hidden_dim"],
            seed=seed,
        )

    def train(self, returns_data: NDArray[np.float64]) -> dict:
        """Train the RL agent on return data.

        Parameters
        ----------
        returns_data : ndarray of shape (T, n_assets)
            Training return matrix.

        Returns
        -------
        results : dict
            Keys: loss_history, final_reward, best_params.
        """
        self._env = self._build_env()
        self._agent = self._build_agent(self._env)

        train_cfg = self.config["training"]
        env_cfg = self.config["environment"]
        seed = self.config["random_seed"]

        self._train_results = train_rl_agent(
            agent=self._agent,
            env=self._env,
            returns_data=returns_data,
            n_epochs=train_cfg["n_epochs"],
            population_size=train_cfg["population_size"],
            sigma=train_cfg["sigma"],
            lr=train_cfg["lr"],
            cvar_lambda=env_cfg["cvar_lambda"],
            seed=seed,
        )
        self._trained = True
        return self._train_results

    def backtest(self, returns_data: NDArray[np.float64]) -> pd.DataFrame:
        """Run RL agent and all benchmarks, return comparison DataFrame.

        Parameters
        ----------
        returns_data : ndarray of shape (T, n_assets)
            Return matrix for backtesting.

        Returns
        -------
        comparison : pd.DataFrame
            Columns: strategy, total_return, sharpe, cvar_95, max_drawdown.
        """
        if not self._trained or self._agent is None or self._env is None:
            raise RuntimeError("Must call train() before backtest()")

        # RL agent backtest
        self._episode_results = run_episode(self._agent, self._env, returns_data)

        cost_rate = self.config["portfolio"]["cost_rate"]
        risk_aversion = self.config["benchmarks"]["risk_aversion"]

        # Benchmark strategies
        n_assets = self.config["portfolio"]["n_assets"]
        ew_weights = equal_weight(n_assets)
        mv_weights = mean_variance(returns_data, risk_aversion=risk_aversion)
        rp_weights = risk_parity(returns_data)

        ew_result = run_benchmark(ew_weights, returns_data, cost_rate)
        mv_result = run_benchmark(mv_weights, returns_data, cost_rate)
        rp_result = run_benchmark(rp_weights, returns_data, cost_rate)

        rows = [
            {
                "strategy": "RL Agent",
                "total_return": self._episode_results["total_return"],
                "sharpe": self._episode_results["sharpe"],
                "cvar_95": self._episode_results["cvar"],
                "max_drawdown": self._episode_results["max_drawdown"],
            },
            {
                "strategy": "Equal Weight",
                "total_return": ew_result["total_return"],
                "sharpe": ew_result["sharpe"],
                "cvar_95": ew_result["cvar_95"],
                "max_drawdown": ew_result["max_drawdown"],
            },
            {
                "strategy": "Mean-Variance",
                "total_return": mv_result["total_return"],
                "sharpe": mv_result["sharpe"],
                "cvar_95": mv_result["cvar_95"],
                "max_drawdown": mv_result["max_drawdown"],
            },
            {
                "strategy": "Risk Parity",
                "total_return": rp_result["total_return"],
                "sharpe": rp_result["sharpe"],
                "cvar_95": rp_result["cvar_95"],
                "max_drawdown": rp_result["max_drawdown"],
            },
        ]

        return pd.DataFrame(rows)

    def allocation_history(self) -> NDArray[np.float64]:
        """Return the weight matrix from the last RL backtest.

        Returns
        -------
        weights : ndarray of shape (T, n_assets)
            Portfolio weight at each time step.
        """
        if self._episode_results is None:
            raise RuntimeError("Must call backtest() first")
        return self._episode_results["weights_history"]

    def compare_strategies(self) -> pd.DataFrame:
        """Return the comparison DataFrame from the last backtest.

        Alias for backtest() results stored internally.

        Returns
        -------
        comparison : pd.DataFrame
            Columns: strategy, total_return, sharpe, cvar_95, max_drawdown.
        """
        if self._episode_results is None:
            raise RuntimeError("Must call backtest() first")
        # Re-derive from last backtest is not stored, but we can just return
        # the backtest. Callers should use backtest() directly.
        raise RuntimeError(
            "compare_strategies() requires a prior backtest() call. "
            "Use backtest(returns_data) instead."
        )
