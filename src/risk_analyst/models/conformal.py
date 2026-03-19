"""Conformal prediction primitives for distribution-free risk intervals.

Implements:
    1. Split conformal prediction -- symmetric intervals from calibration residuals
    2. Conformalized quantile regression (CQR) -- asymmetric intervals
    3. Conformal risk control (CRC) -- monotone risk functional thresholding
    4. Adaptive conformal inference (ACI) -- online coverage under distribution shift

References:
    - Vovk et al. (2005), *Algorithmic Learning in a Random World*.
    - Romano et al. (2019), Conformalized quantile regression.
    - Angelopoulos et al. (2024), Conformal risk control, JMLR 25(332).
    - Gibbs & Candes (2021), Adaptive conformal inference under distribution shift.
"""

from __future__ import annotations

import numpy as np


def split_conformal_threshold(scores_cal: np.ndarray, alpha: float) -> float:
    """Compute the split conformal quantile threshold.

    The threshold is the ceil((1-alpha)(1+1/n))-th quantile of the
    calibration nonconformity scores, guaranteeing marginal coverage
    >= 1 - alpha on exchangeable test data.

    Parameters
    ----------
    scores_cal : np.ndarray
        1-D array of calibration nonconformity scores (e.g. |y - y_hat|).
    alpha : float
        Miscoverage level (e.g. 0.1 for 90% coverage).

    Returns
    -------
    float
        Conformal quantile threshold q_hat.
    """
    n = len(scores_cal)
    quantile_level = min((1 - alpha) * (1 + 1 / n), 1.0)
    return float(np.quantile(scores_cal, quantile_level, method="higher"))


def conformal_prediction_interval(
    predictions: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric conformal prediction interval.

    Constructs [y_hat - q, y_hat + q] for each prediction.

    Parameters
    ----------
    predictions : np.ndarray
        1-D array of point predictions.
    threshold : float
        Conformal threshold from ``split_conformal_threshold``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lower, upper) bounds arrays.
    """
    lower = predictions - threshold
    upper = predictions + threshold
    return lower, upper


def cqr_threshold(
    lower_cal: np.ndarray,
    upper_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float,
) -> float:
    """Conformalized quantile regression (CQR) threshold.

    The nonconformity score is s_i = max(lower_i - y_i, y_i - upper_i),
    measuring how far the true value falls outside the predicted quantile
    interval. The threshold is the conformal quantile of these scores.

    Parameters
    ----------
    lower_cal : np.ndarray
        Lower quantile predictions on calibration set.
    upper_cal : np.ndarray
        Upper quantile predictions on calibration set.
    y_cal : np.ndarray
        True response values on calibration set.
    alpha : float
        Miscoverage level.

    Returns
    -------
    float
        CQR conformal threshold.
    """
    scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
    n = len(scores)
    quantile_level = min((1 - alpha) * (1 + 1 / n), 1.0)
    return float(np.quantile(scores, quantile_level, method="higher"))


def cqr_interval(
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CQR prediction interval: [lower - q, upper + q].

    Parameters
    ----------
    lower_pred : np.ndarray
        Lower quantile predictions on test set.
    upper_pred : np.ndarray
        Upper quantile predictions on test set.
    threshold : float
        CQR threshold from ``cqr_threshold``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lower, upper) bounds arrays.
    """
    lower = lower_pred - threshold
    upper = upper_pred + threshold
    return lower, upper


def conformal_risk_control(
    risk_fn: callable,
    scores_cal: np.ndarray,
    alpha: float,
    lambdas: np.ndarray,
) -> float:
    """Conformal risk control: find the smallest lambda with risk <= alpha.

    Searches over a grid of lambda values (assumed sorted ascending) and
    returns the smallest lambda such that the empirical risk on calibration
    data is at most alpha, with a finite-sample correction.

    Parameters
    ----------
    risk_fn : callable
        Function mapping (scores_cal, lambda) -> float, the empirical risk.
    scores_cal : np.ndarray
        Calibration nonconformity scores.
    alpha : float
        Target risk level.
    lambdas : np.ndarray
        Sorted grid of candidate lambda values (ascending).

    Returns
    -------
    float
        Selected lambda value.
    """
    n = len(scores_cal)
    adjusted_alpha = alpha - 1 / (n + 1)  # finite-sample correction

    for lam in lambdas:
        risk = risk_fn(scores_cal, lam)
        if risk <= max(adjusted_alpha, 0.0):
            return float(lam)

    # If no lambda satisfies the constraint, return the largest
    return float(lambdas[-1])


def adaptive_conformal_update(
    alpha_t: float,
    covered: bool,
    alpha_target: float,
    gamma: float,
) -> float:
    """Single step of Adaptive Conformal Inference (ACI).

    Updates the running miscoverage level:
        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

    where err_t = 1 if not covered, 0 if covered.

    When the observation is NOT covered (err_t=1 > alpha_target), alpha_t
    decreases, which raises the quantile level for the next threshold,
    producing wider intervals. When covered (err_t=0 < alpha_target),
    alpha_t increases, lowering the quantile level and tightening intervals.

    Parameters
    ----------
    alpha_t : float
        Current adaptive miscoverage level.
    covered : bool
        Whether the latest observation was covered by the interval.
    alpha_target : float
        Target miscoverage level (e.g. 0.1 for 90% coverage).
    gamma : float
        Step size controlling adaptation speed.

    Returns
    -------
    float
        Updated alpha_{t+1}, clipped to [0, 1].
    """
    err_t = 1.0 - float(covered)
    alpha_new = alpha_t + gamma * (alpha_target - err_t)
    return float(np.clip(alpha_new, 0.0, 1.0))
