"""Sobol global sensitivity analysis for climate risk factors.

Implements first-order (S1) and total-order (ST) Sobol sensitivity indices
using either OpenTURNS (preferred) or manual Saltelli sampling.

References:
    - Sobol' (1993). Sensitivity estimates for nonlinear mathematical models.
    - Saltelli (2002). Making best use of model evaluations to compute
      sensitivity indices.
    - Saltelli et al. (2010). Variance based sensitivity analysis of
      model output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Manual Saltelli sampling
# ---------------------------------------------------------------------------

def saltelli_sample(
    n: int,
    d: int,
    bounds: list[tuple],
    seed: int = 42,
) -> np.ndarray:
    """Generate Saltelli sampling design for Sobol analysis.

    The Saltelli design requires n * (2*d + 2) model evaluations to
    compute both first-order and total-order indices.

    Parameters
    ----------
    n : int
        Base sample size.
    d : int
        Number of input factors.
    bounds : list[tuple]
        List of (lower, upper) bounds for each factor.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Sampling matrix of shape (n * (2*d + 2), d).
    """
    rng = np.random.default_rng(seed)

    # Generate two independent quasi-random matrices A and B
    A = rng.random((n, d))
    B = rng.random((n, d))

    # Scale to bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    A = A * (upper - lower) + lower
    B = B * (upper - lower) + lower

    # Build AB_i matrices: A with column i replaced by B's column i
    # and BA_i matrices: B with column i replaced by A's column i
    samples = [A, B]
    for i in range(d):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        samples.append(AB_i)
    for i in range(d):
        BA_i = B.copy()
        BA_i[:, i] = A[:, i]
        samples.append(BA_i)

    return np.vstack(samples)


def sobol_indices(
    Y_A: np.ndarray,
    Y_B: np.ndarray,
    Y_AB_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first-order (S1) and total-order (ST) Sobol indices.

    Uses the Jansen (1999) estimator for ST and the Saltelli (2010)
    estimator for S1.

    Parameters
    ----------
    Y_A : np.ndarray
        Model output evaluated on matrix A, shape (n,).
    Y_B : np.ndarray
        Model output evaluated on matrix B, shape (n,).
    Y_AB_list : list[np.ndarray]
        List of d arrays, each shape (n,). Y_AB_list[i] is the model
        output evaluated on AB_i (A with column i from B).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (S1, ST) arrays of shape (d,).
    """
    n = len(Y_A)
    d = len(Y_AB_list)
    f0 = np.mean(np.concatenate([Y_A, Y_B]))
    var_total = np.var(np.concatenate([Y_A, Y_B]))

    if var_total < 1e-15:
        return np.zeros(d), np.zeros(d)

    S1 = np.zeros(d)
    ST = np.zeros(d)

    for i in range(d):
        Y_ABi = Y_AB_list[i]

        # First-order: Saltelli (2010) estimator
        # V_i = (1/n) * sum(Y_B * (Y_AB_i - Y_A))
        V_i = np.mean(Y_B * (Y_ABi - Y_A))
        S1[i] = V_i / var_total

        # Total-order: Jansen (1999) estimator
        # E_i = (1/(2n)) * sum((Y_A - Y_AB_i)^2)
        E_i = 0.5 * np.mean((Y_A - Y_ABi) ** 2)
        ST[i] = E_i / var_total

    # Clip to valid range: S1 and ST in [0, 1], and enforce ST >= S1
    S1 = np.clip(S1, 0.0, 1.0)
    ST = np.clip(ST, 0.0, 1.0)
    ST = np.maximum(ST, S1)

    return S1, ST


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sobol_analysis(
    model_fn: callable,
    factor_names: list[str],
    bounds: list[tuple],
    n_samples: int = 1024,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Sobol sensitivity analysis on a model function.

    Tries OpenTURNS SobolIndicesAlgorithm first; falls back to manual
    Saltelli sampling if OpenTURNS is not installed.

    Parameters
    ----------
    model_fn : callable
        Function mapping np.ndarray of shape (d,) -> float.
    factor_names : list[str]
        Names of the d input factors.
    bounds : list[tuple]
        List of (lower, upper) bounds for each factor.
    n_samples : int
        Base sample size (total evaluations = n * (2*d + 2)).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: factor, S1, ST.
    """
    d = len(factor_names)

    # Try OpenTURNS first
    try:
        import openturns as ot  # type: ignore[import-untyped]
        return _sobol_openturns(model_fn, factor_names, bounds, n_samples, seed)
    except ImportError:
        pass

    # Fallback: manual Saltelli
    return _sobol_manual(model_fn, factor_names, bounds, n_samples, seed)


def _sobol_openturns(
    model_fn: callable,
    factor_names: list[str],
    bounds: list[tuple],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Sobol analysis using OpenTURNS."""
    import openturns as ot  # type: ignore[import-untyped]

    d = len(factor_names)

    # Define input distribution (uniform on bounds)
    marginals = [ot.Uniform(float(lo), float(hi)) for lo, hi in bounds]
    distribution = ot.JointDistribution(marginals)

    # Wrap model function
    class OTModel(ot.OpenTURNSPythonFunction):
        def __init__(self) -> None:
            super().__init__(d, 1)

        def _exec(self, X: list) -> list:
            x = np.array(X)
            return [float(model_fn(x))]

    ot_model = ot.Function(OTModel())

    # Sobol experiment
    ot.RandomGenerator.SetSeed(seed)
    size = n_samples
    experiment = ot.SobolIndicesExperiment(distribution, size)
    input_design = experiment.generate()
    output_design = ot_model(input_design)

    # Compute indices
    algo = ot.SaltelliSensitivityAlgorithm(
        input_design, output_design, size
    )

    s1 = np.array([algo.getFirstOrderIndices()[i] for i in range(d)])
    st = np.array([algo.getTotalOrderIndices()[i] for i in range(d)])

    s1 = np.clip(s1, 0.0, None)

    return pd.DataFrame({
        "factor": factor_names,
        "S1": s1,
        "ST": st,
    })


def _sobol_manual(
    model_fn: callable,
    factor_names: list[str],
    bounds: list[tuple],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Sobol analysis using manual Saltelli sampling."""
    d = len(factor_names)
    n = n_samples

    # Generate Saltelli sample
    sample_matrix = saltelli_sample(n, d, bounds, seed=seed)

    # Evaluate model on all samples
    Y_all = np.array([model_fn(sample_matrix[i]) for i in range(len(sample_matrix))])

    # Split outputs: A, B, AB_1..AB_d, BA_1..BA_d
    Y_A = Y_all[:n]
    Y_B = Y_all[n:2 * n]
    Y_AB_list = [Y_all[(2 + i) * n:(3 + i) * n] for i in range(d)]

    # Compute indices
    S1, ST = sobol_indices(Y_A, Y_B, Y_AB_list)

    return pd.DataFrame({
        "factor": factor_names,
        "S1": S1,
        "ST": ST,
    })
