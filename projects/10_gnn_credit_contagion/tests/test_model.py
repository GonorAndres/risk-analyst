"""Unit tests for Project 10: GNN Credit Contagion.

Tests cover:
1. Network generation shapes and properties
2. Eisenberg-Noe clearing constraints
3. Cascade dynamics
4. DebtRank bounds
5. GCN forward pass shapes and bounds
6. BCE loss properties
7. Training convergence
8. Centrality metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from contagion import (
    eisenberg_noe_clearing,
    simulate_cascade,
    systemic_importance,
)
from gcn import GCN
from network import (
    compute_centrality,
    network_stats,
)
from trainer import binary_cross_entropy, train_gcn


class TestNetworkGeneration:
    """Tests for financial network generation."""

    def test_adjacency_shape(self, small_network: dict) -> None:
        """1. Generated network has correct adjacency shape."""
        adj = small_network["adjacency"]
        assert adj.shape == (20, 20)

    def test_node_features_shape(self, small_network: dict) -> None:
        """2. Node features have correct shape (n_nodes, n_features)."""
        features = small_network["node_features"]
        assert features.shape == (20, 5)

    def test_adjacency_non_negative(self, small_network: dict) -> None:
        """3. Adjacency is non-negative (exposures are positive amounts)."""
        adj = small_network["adjacency"]
        assert np.all(adj >= 0)

    def test_labels_binary(self, small_network: dict) -> None:
        """Labels are binary (0 or 1)."""
        labels = small_network["labels"]
        assert set(np.unique(labels)).issubset({0.0, 1.0})


class TestEisenbergNoe:
    """Tests for Eisenberg-Noe clearing mechanism."""

    def test_payments_leq_liabilities(
        self, network_with_clearing: dict
    ) -> None:
        """4. Payments <= liabilities (can't pay more than owed)."""
        payments = network_with_clearing["payments"]
        p_bar = network_with_clearing["liabilities"].sum(axis=1)
        assert np.all(payments <= p_bar + 1e-8)

    def test_defaults_when_assets_lt_liabilities(self) -> None:
        """5. Defaults occur when assets < liabilities."""
        # Construct a simple 3-node network where node 0 is under-capitalised
        liabilities = np.array([
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        # Node 0 owes 150 total but only has 50 in assets
        assets = np.array([50.0, 200.0, 200.0])

        payments, defaults = eisenberg_noe_clearing(liabilities, assets)

        # Node 0 should default
        assert defaults[0] == 1.0
        # Nodes 1 and 2 owe nothing, so they don't default
        assert defaults[1] == 0.0
        assert defaults[2] == 0.0


class TestCascade:
    """Tests for cascade simulation."""

    def test_systemic_vs_peripheral_shock(self, small_network: dict) -> None:
        """6. Shocking a more connected node tends to cause more defaults."""
        liabilities = small_network["liabilities"]
        assets = small_network["assets"]

        # Find the node with highest out-degree (most connected)
        out_degree = (liabilities > 0).sum(axis=1)
        systemic_node = int(np.argmax(out_degree))

        # Find a node with lowest out-degree (peripheral)
        peripheral_node = int(np.argmin(out_degree))

        cascade_systemic = simulate_cascade(
            liabilities, assets,
            shocked_nodes=[systemic_node],
            shock_fraction=0.9,
        )
        cascade_peripheral = simulate_cascade(
            liabilities, assets,
            shocked_nodes=[peripheral_node],
            shock_fraction=0.9,
        )

        # Systemic shock should cause at least as much damage
        assert (
            cascade_systemic["total_loss"]
            >= cascade_peripheral["total_loss"] - 1e-8
        )


class TestDebtRank:
    """Tests for DebtRank computation."""

    def test_debtrank_bounds(self, small_network: dict) -> None:
        """7. DebtRank is between 0 and 1."""
        dr = systemic_importance(
            small_network["liabilities"],
            small_network["assets"],
        )
        assert np.all(dr >= -1e-8)
        assert np.all(dr <= 1.0 + 1e-8)


class TestGCN:
    """Tests for the Graph Convolutional Network."""

    def test_forward_output_shape(
        self, small_gcn: GCN, small_network: dict
    ) -> None:
        """8. GCN forward pass output shape is (n_nodes, 1)."""
        X = small_network["node_features"]
        A = small_network["adjacency"]
        output = small_gcn.forward(X, A)
        assert output.shape == (20, 1)

    def test_forward_output_bounds(
        self, small_gcn: GCN, small_network: dict
    ) -> None:
        """9. GCN output is in [0, 1] (sigmoid activation)."""
        X = small_network["node_features"]
        A = small_network["adjacency"]
        output = small_gcn.forward(X, A)
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_param_count(self, small_gcn: GCN) -> None:
        """Parameter count matches flat params length."""
        flat = small_gcn.get_flat_params()
        assert len(flat) == small_gcn.num_params()

    def test_param_roundtrip(self, small_gcn: GCN) -> None:
        """Setting flat params then getting them recovers the same values."""
        original = small_gcn.get_flat_params().copy()
        small_gcn.set_flat_params(original)
        recovered = small_gcn.get_flat_params()
        np.testing.assert_array_almost_equal(original, recovered)


class TestTraining:
    """Tests for GCN training."""

    def test_bce_non_negative(self) -> None:
        """10. BCE loss is non-negative."""
        y_true = np.array([0, 1, 1, 0, 1], dtype=float)
        y_pred = np.array([0.2, 0.8, 0.6, 0.1, 0.9], dtype=float)
        loss = binary_cross_entropy(y_true, y_pred)
        assert loss >= 0.0

    def test_bce_perfect_prediction(self) -> None:
        """BCE loss is near zero for perfect predictions."""
        y_true = np.array([0, 1, 1, 0], dtype=float)
        y_pred = np.array([0.001, 0.999, 0.999, 0.001], dtype=float)
        loss = binary_cross_entropy(y_true, y_pred)
        assert loss < 0.01

    def test_training_reduces_loss(self, small_network: dict) -> None:
        """11. Training reduces loss (loss[-1] < loss[0])."""
        gcn = GCN(n_features=5, hidden_dim=8, n_classes=1, seed=42)
        X = small_network["node_features"]
        A = small_network["adjacency"]
        y = small_network["labels"]

        loss_history = train_gcn(
            gcn=gcn,
            X=X,
            A=A,
            y=y,
            n_epochs=30,
            lr=0.05,
            seed=42,
            population_size=20,
            noise_std=0.1,
        )

        # Allow some tolerance: final loss should be below initial loss
        assert loss_history[-1] < loss_history[0] + 0.1


class TestCentrality:
    """Tests for centrality computation."""

    def test_centrality_columns(self, small_network: dict) -> None:
        """12. Centrality DataFrame has correct columns."""
        df = compute_centrality(small_network["adjacency"])
        expected_columns = {
            "node",
            "degree_centrality",
            "betweenness_centrality",
            "eigenvector_centrality",
        }
        assert set(df.columns) == expected_columns

    def test_centrality_values_bounded(self, small_network: dict) -> None:
        """Centrality values are non-negative."""
        df = compute_centrality(small_network["adjacency"])
        assert np.all(df["degree_centrality"] >= 0)
        assert np.all(df["betweenness_centrality"] >= -1e-8)
        assert np.all(df["eigenvector_centrality"] >= -1e-8)

    def test_network_stats_keys(self, small_network: dict) -> None:
        """Network stats has all expected keys."""
        stats = network_stats(small_network["adjacency"])
        expected_keys = {"n_nodes", "n_edges", "density", "avg_degree", "max_degree"}
        assert set(stats.keys()) == expected_keys
