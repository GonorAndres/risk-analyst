"""Shared fixtures for Project 11 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path so imports work without installation
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)


@pytest.fixture
def seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(seed)


@pytest.fixture
def regression_data(rng: np.random.Generator) -> dict:
    """Exchangeable regression data for conformal tests."""
    X = rng.standard_normal((1000, 5))
    y = X @ np.array([1, -0.5, 0.3, 0, 0.2]) + rng.standard_normal(1000) * 0.5
    return {"X": X, "y": y}


@pytest.fixture
def credit_data(rng: np.random.Generator) -> dict:
    """Synthetic credit data with logistic default probabilities."""
    n = 2000
    X_credit = rng.standard_normal((n, 5))
    logits = X_credit @ np.array([0.5, -0.3, 0.2, 0.1, -0.4]) - 1.5
    probs = 1 / (1 + np.exp(-logits))
    y_credit = rng.binomial(1, probs)
    return {"X": X_credit, "y": y_credit, "probs": probs}


@pytest.fixture
def config() -> dict:
    """Default config dict for tests."""
    return {
        "data": {"n_samples": 3000, "n_features": 5, "seed": 42},
        "conformal": {"alpha": 0.1, "cal_fraction": 0.3},
        "cqr": {"n_estimators": 100, "max_depth": 4},
        "adaptive": {"gamma": 0.01, "n_calm": 500, "n_crisis": 300},
        "credit": {"n_samples": 5000, "default_rate": 0.10},
        "random_seed": 42,
    }
