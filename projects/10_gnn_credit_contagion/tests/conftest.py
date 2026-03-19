"""Shared fixtures for Project 10 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path so imports work without installation
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from contagion import eisenberg_noe_clearing
from gcn import GCN
from network import generate_financial_network


@pytest.fixture
def seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def small_network(seed: int) -> dict:
    """Small financial network for fast tests."""
    return generate_financial_network(n_nodes=20, n_edges=60, seed=seed)


@pytest.fixture
def medium_network(seed: int) -> dict:
    """Medium financial network matching default config."""
    return generate_financial_network(n_nodes=50, n_edges=150, seed=seed)


@pytest.fixture
def small_gcn(seed: int) -> GCN:
    """Small GCN for testing."""
    return GCN(n_features=5, hidden_dim=8, n_classes=1, seed=seed)


@pytest.fixture
def network_with_clearing(small_network: dict) -> dict:
    """Small network with Eisenberg-Noe clearing results."""
    payments, defaults = eisenberg_noe_clearing(
        small_network["liabilities"],
        small_network["assets"],
    )
    small_network["payments"] = payments
    small_network["clearing_defaults"] = defaults
    return small_network
