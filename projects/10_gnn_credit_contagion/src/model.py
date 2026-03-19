"""High-level model orchestrating network, contagion, and GCN training.

GNNContagionModel ties together all components:
1. Generate a synthetic financial network
2. Run Eisenberg-Noe cascade to derive default labels
3. Train a GCN to predict default probabilities from network structure
4. Analyse systemic risk using DebtRank and centrality measures

Reference:
- Guo et al. (2025), "Credit risk contagion modeling using GNNs", ACM BAIDE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from contagion import (
    eisenberg_noe_clearing,
    simulate_cascade,
    systemic_importance,
)
from gcn import GCN
from network import (
    compute_centrality,
    generate_financial_network,
    network_stats,
)
from trainer import evaluate_gcn, train_gcn


def _load_config(config_path: str | None = None) -> dict:
    """Load YAML configuration.

    Parameters
    ----------
    config_path : str or None
        Path to YAML config. Defaults to configs/default.yaml.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / "configs" / "default.yaml"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class GNNContagionModel:
    """End-to-end model for GNN-based credit contagion analysis.

    Parameters
    ----------
    config : dict
        Configuration with keys: network, contagion, gcn, training, random_seed.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.seed = config.get("random_seed", 42)
        self.network_data: dict | None = None
        self.gcn: GCN | None = None
        self.loss_history: list[float] | None = None

    def build_network(self) -> dict:
        """Generate financial network and compute cascade-based labels.

        Generates the network, then runs Eisenberg-Noe clearing to
        determine which nodes default in equilibrium. These equilibrium
        defaults become the training labels for the GCN.

        Returns
        -------
        dict
            Network data with adjacency, node_features, liabilities,
            assets, labels, and stats.
        """
        net_cfg = self.config["network"]
        self.network_data = generate_financial_network(
            n_nodes=net_cfg["n_nodes"],
            n_edges=net_cfg["n_edges"],
            seed=self.seed,
        )

        # Refine labels using Eisenberg-Noe clearing
        payments, defaults = eisenberg_noe_clearing(
            self.network_data["liabilities"],
            self.network_data["assets"],
        )
        # Combine initial capital-ratio labels with clearing defaults
        combined_labels = np.maximum(
            self.network_data["labels"], defaults
        )
        self.network_data["labels"] = combined_labels

        # Add network statistics
        self.network_data["stats"] = network_stats(
            self.network_data["adjacency"]
        )

        return self.network_data

    def train(self) -> dict:
        """Train the GCN on the financial network.

        Returns
        -------
        dict
            Training results: loss_history, metrics (accuracy, f1, etc.),
            n_params, n_epochs.
        """
        if self.network_data is None:
            self.build_network()

        gcn_cfg = self.config["gcn"]
        train_cfg = self.config["training"]
        net_cfg = self.config["network"]

        self.gcn = GCN(
            n_features=net_cfg["n_features"],
            hidden_dim=gcn_cfg["hidden_dim"],
            n_classes=gcn_cfg["n_classes"],
            seed=self.seed,
        )

        self.loss_history = train_gcn(
            gcn=self.gcn,
            X=self.network_data["node_features"],
            A=self.network_data["adjacency"],
            y=self.network_data["labels"],
            n_epochs=train_cfg["n_epochs"],
            lr=train_cfg["lr"],
            seed=self.seed,
            population_size=train_cfg.get("population_size", 40),
        )

        metrics = evaluate_gcn(
            gcn=self.gcn,
            X=self.network_data["node_features"],
            A=self.network_data["adjacency"],
            y=self.network_data["labels"],
        )

        return {
            "loss_history": self.loss_history,
            "metrics": metrics,
            "n_params": self.gcn.num_params(),
            "n_epochs": train_cfg["n_epochs"],
        }

    def analyze_systemic_risk(self) -> pd.DataFrame:
        """Combine DebtRank, centrality, and GCN predictions per node.

        Returns
        -------
        pd.DataFrame
            Columns: node, debtrank, degree_centrality, betweenness_centrality,
                     eigenvector_centrality, gcn_default_prob, true_label.
        """
        if self.network_data is None:
            self.build_network()

        # DebtRank
        dr = systemic_importance(
            self.network_data["liabilities"],
            self.network_data["assets"],
        )

        # Centrality
        centrality_df = compute_centrality(self.network_data["adjacency"])

        # GCN predictions (if trained)
        if self.gcn is not None:
            gcn_pred = self.gcn.forward(
                self.network_data["node_features"],
                self.network_data["adjacency"],
            ).ravel()
        else:
            gcn_pred = np.full(len(dr), np.nan)

        result = centrality_df.copy()
        result["debtrank"] = dr
        result["gcn_default_prob"] = gcn_pred
        result["true_label"] = self.network_data["labels"]

        return result

    def cascade_analysis(self, n_simulations: int | None = None) -> pd.DataFrame:
        """Shock each node individually and measure cascade size.

        Parameters
        ----------
        n_simulations : int or None
            Number of nodes to shock. Defaults to config value or all nodes.

        Returns
        -------
        pd.DataFrame
            Columns: shocked_node, n_defaults, total_loss, rounds.
        """
        if self.network_data is None:
            self.build_network()

        cont_cfg = self.config["contagion"]
        shock_fraction = cont_cfg["shock_fraction"]

        n_nodes = self.network_data["adjacency"].shape[0]
        if n_simulations is None:
            n_simulations = min(
                cont_cfg.get("n_cascade_simulations", n_nodes),
                n_nodes,
            )

        results = []
        for node in range(n_simulations):
            cascade = simulate_cascade(
                self.network_data["liabilities"],
                self.network_data["assets"],
                shocked_nodes=[node],
                shock_fraction=shock_fraction,
            )
            results.append({
                "shocked_node": node,
                "n_defaults": cascade["n_defaults"],
                "total_loss": cascade["total_loss"],
                "rounds": cascade["rounds"],
            })

        return pd.DataFrame(results)
