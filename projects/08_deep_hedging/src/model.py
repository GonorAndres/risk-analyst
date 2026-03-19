"""High-level deep hedging model orchestration.

Brings together environment, network, and trainer to run end-to-end
deep hedging experiments and compare with Black-Scholes benchmarks.

Reference: Buehler et al. (2019), "Deep hedging", Quantitative Finance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from environment import HedgingEnvironment
from network import NeuralNetwork
from trainer import DeepHedgingTrainer

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class DeepHedgingModel:
    """Orchestrates deep hedging training, evaluation, and benchmarking.

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
        self._network: NeuralNetwork | None = None
        self._env_train: HedgingEnvironment | None = None
        self._env_test: HedgingEnvironment | None = None
        self._trainer: DeepHedgingTrainer | None = None
        self._loss_history: list[float] = []

    def _build_env(self, n_paths: int, seed: int) -> HedgingEnvironment:
        """Create a HedgingEnvironment from config."""
        opt = self.config["option"]
        return HedgingEnvironment(
            s0=opt["s0"],
            K=opt["K"],
            r=opt["r"],
            sigma=opt["sigma"],
            T=opt["T"],
            n_steps=opt["n_steps"],
            n_paths=n_paths,
            cost_rate=self.config["transaction_cost"]["rate"],
            seed=seed,
        )

    def train_hedger(self) -> dict:
        """Train the deep hedging network.

        Sets up the environment, network, and trainer, then trains
        and returns a results dictionary.

        Returns
        -------
        results : dict
            Keys: loss_history, final_risk, trained_positions (on test set).
        """
        seed = self.config["random_seed"]
        sim = self.config["simulation"]
        net_cfg = self.config["network"]
        train_cfg = self.config["training"]

        # Build training environment
        self._env_train = self._build_env(sim["n_paths_train"], seed)
        self._env_test = self._build_env(sim["n_paths_test"], seed + 1)

        # Build network
        self._network = NeuralNetwork(
            layer_sizes=net_cfg["layer_sizes"], seed=seed
        )

        # Build trainer
        trainer_config = {
            "population_size": train_cfg["population_size"],
            "lr": train_cfg["lr"],
            "risk_measure": train_cfg["risk_measure"],
            "cvar_alpha": train_cfg.get("cvar_alpha", 0.95),
        }
        self._trainer = DeepHedgingTrainer(
            env=self._env_train,
            network=self._network,
            config=trainer_config,
        )

        # Train
        self._loss_history = self._trainer.train(
            n_epochs=train_cfg["n_epochs"],
            lr=train_cfg["lr"],
            measure=train_cfg["risk_measure"],
        )
        self._trained = True

        # Evaluate on test set
        test_paths = self._env_test.simulate_paths()
        positions = self._trainer.compute_hedge_positions(test_paths)
        pnl = self._env_test.compute_pnl(test_paths, positions)
        final_risk = self._trainer.risk_measure(pnl, train_cfg["risk_measure"])

        return {
            "loss_history": self._loss_history,
            "final_risk": final_risk,
            "trained_positions": positions,
        }

    def compare_with_bs(self) -> pd.DataFrame:
        """Compare BS delta hedging vs deep hedging on the test set.

        Returns
        -------
        comparison : DataFrame
            Columns: method, mean_pnl, std_pnl, cvar_95, max_loss.
        """
        if not self._trained:
            self.train_hedger()

        assert self._env_test is not None
        assert self._trainer is not None

        test_paths = self._env_test.simulate_paths()
        n_paths = test_paths.shape[0]
        n_steps = test_paths.shape[1] - 1

        # --- Black-Scholes delta hedging ---
        bs_positions = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            time_t = t * self._env_test.dt
            bs_positions[:, t] = self._env_test.bs_delta(
                test_paths[:, t], time_t
            )
        bs_pnl = self._env_test.compute_pnl(test_paths, bs_positions)

        # --- Deep hedging ---
        deep_positions = self._trainer.compute_hedge_positions(test_paths)
        deep_pnl = self._env_test.compute_pnl(test_paths, deep_positions)

        # --- No hedging ---
        no_hedge_positions = np.zeros((n_paths, n_steps))
        premium = float(self._env_test.bs_price(self._env_test.s0, 0.0))
        no_hedge_pnl = premium - self._env_test.compute_payoff(test_paths[:, -1])

        # Compute metrics
        rows = []
        for name, pnl in [
            ("No Hedge", no_hedge_pnl),
            ("BS Delta", bs_pnl),
            ("Deep Hedge", deep_pnl),
        ]:
            losses = -pnl
            cvar_95 = float(np.mean(losses[losses >= np.percentile(losses, 95)]))
            rows.append(
                {
                    "method": name,
                    "mean_pnl": float(np.mean(pnl)),
                    "std_pnl": float(np.std(pnl)),
                    "cvar_95": cvar_95,
                    "max_loss": float(np.max(losses)),
                }
            )

        return pd.DataFrame(rows)

    def analyze_positions(self) -> pd.DataFrame:
        """Compare learned hedge ratios vs BS delta at various (S, t) points.

        Returns
        -------
        analysis : DataFrame
            Columns: S, t, bs_delta, deep_delta, difference.
        """
        if not self._trained:
            self.train_hedger()

        assert self._env_test is not None
        assert self._network is not None

        s0 = self._env_test.s0
        T = self._env_test.T

        # Grid of (S, t) points
        S_values = np.linspace(s0 * 0.8, s0 * 1.2, 11)
        t_values = np.array([0.0, T * 0.25, T * 0.5, T * 0.75])

        rows = []
        for t in t_values:
            for S in S_values:
                bs_d = float(self._env_test.bs_delta(S, t))

                # Network prediction
                features = np.array([[S / s0, t / T, bs_d]])  # use bs_delta as proxy position
                deep_d = float(self._network.forward(features).ravel()[0])

                rows.append(
                    {
                        "S": S,
                        "t": round(t, 6),
                        "bs_delta": round(bs_d, 6),
                        "deep_delta": round(deep_d, 6),
                        "difference": round(deep_d - bs_d, 6),
                    }
                )

        return pd.DataFrame(rows)
