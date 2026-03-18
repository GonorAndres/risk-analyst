"""Reverse stress testing and sensitivity analysis.

Finds the minimum-norm macro shock vector that causes portfolio losses to
exceed a given threshold, and provides local sensitivity diagnostics.

References:
    - Breuer & Csiszar (2013), Systematic stress tests with entropic
      plausibility constraints, *J. Banking & Finance* 37(5), 1552--1559.
    - Basel Committee (2018), Stress Testing Principles, Principle 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from transmission import MacroTransmissionModel


def reverse_stress_test(
    model: MacroTransmissionModel,
    loss_threshold: float,
    factor_names: list[str],
    bounds: list[tuple[float, float]] | None = None,
) -> dict:
    """Find the minimum-norm shock vector whose predicted loss >= *loss_threshold*.

    Solves::

        min  ||x||^2
        s.t. predict_loss(x) >= loss_threshold

    using SLSQP.

    Parameters
    ----------
    model : MacroTransmissionModel
        A fitted transmission model.
    loss_threshold : float
        Target minimum loss (e.g. 0.15 for 15%).
    factor_names : list[str]
        Factor names corresponding to the model's factor ordering.
    bounds : list[tuple[float, float]] | None
        Per-factor (lower, upper) bounds.  Defaults to (-1, 1) for each.

    Returns
    -------
    dict
        Keys: ``optimal_shocks`` (dict), ``predicted_loss`` (float),
        ``shock_norm`` (float), ``success`` (bool).
    """
    k = len(factor_names)
    if bounds is None:
        bounds = [(-1.0, 1.0)] * k

    def objective(x: np.ndarray) -> float:
        """Minimise L2 norm of shock vector."""
        return float(x @ x)

    def constraint_loss(x: np.ndarray) -> float:
        """predict_loss(x) - threshold >= 0."""
        shocks = {name: x[i] for i, name in enumerate(factor_names)}
        return model.predict_loss(shocks) - loss_threshold

    constraints = [{"type": "ineq", "fun": constraint_loss}]
    x0 = np.zeros(k)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    optimal_shocks = {name: float(result.x[i]) for i, name in enumerate(factor_names)}
    predicted_loss = model.predict_loss(optimal_shocks)

    return {
        "optimal_shocks": optimal_shocks,
        "predicted_loss": predicted_loss,
        "shock_norm": float(np.linalg.norm(result.x)),
        "success": bool(result.success and predicted_loss >= loss_threshold * 0.99),
    }


def sensitivity_analysis(
    model: MacroTransmissionModel,
    base_shocks: dict[str, float],
    factor_names: list[str],
    perturbation: float = 0.01,
) -> pd.DataFrame:
    """Measure loss sensitivity to small perturbations of each factor.

    For each factor, bump the shock by +/- *perturbation* and record the
    change in predicted loss to compute a finite-difference elasticity.

    Parameters
    ----------
    model : MacroTransmissionModel
        Fitted transmission model.
    base_shocks : dict[str, float]
        Base scenario shock vector.
    factor_names : list[str]
        Factor names to perturb.
    perturbation : float
        Absolute perturbation size (default 0.01).

    Returns
    -------
    pd.DataFrame
        Columns: factor, base_loss, perturbed_loss, elasticity.
    """
    base_loss = model.predict_loss(base_shocks)
    rows: list[dict] = []

    for name in factor_names:
        perturbed = base_shocks.copy()
        perturbed[name] = perturbed.get(name, 0.0) + perturbation

        perturbed_loss = model.predict_loss(perturbed)
        delta = perturbed_loss - base_loss
        elasticity = delta / perturbation if perturbation != 0 else 0.0

        rows.append({
            "factor": name,
            "base_loss": base_loss,
            "perturbed_loss": perturbed_loss,
            "elasticity": elasticity,
        })

    return pd.DataFrame(rows)
