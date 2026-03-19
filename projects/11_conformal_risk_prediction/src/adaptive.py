"""Adaptive Conformal Inference (ACI) for non-stationary environments.

Implements online conformal prediction that adapts coverage under
distributional shift (e.g. volatility regime changes in financial data).

References:
    - Gibbs & Candes (2021), Adaptive conformal inference under
      distribution shift, NeurIPS 2021.
    - Zaffran et al. (2022), Adaptive conformal predictions for time series.
"""

from __future__ import annotations

import numpy as np

from risk_analyst.models.conformal import (
    adaptive_conformal_update,
    split_conformal_threshold,
)


class AdaptiveConformalInference:
    """Online adaptive conformal inference with coverage tracking.

    Dynamically adjusts the miscoverage level alpha_t so that long-run
    coverage converges to the target (1 - alpha) even when the data
    distribution shifts over time.

    Parameters
    ----------
    alpha : float
        Target miscoverage level (e.g. 0.1 for 90% coverage).
    gamma : float
        Step size controlling how quickly the threshold adapts.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01) -> None:
        self.alpha_target = alpha
        self.gamma = gamma
        self._alpha_t: float = alpha
        self._coverage_history: list[bool] = []
        self._alpha_history: list[float] = [alpha]

    @property
    def alpha_t(self) -> float:
        """Current adaptive miscoverage level."""
        return self._alpha_t

    def update(self, score: float, threshold: float) -> None:
        """Update alpha_t based on whether the score is covered.

        Parameters
        ----------
        score : float
            Nonconformity score of the latest observation.
        threshold : float
            Current conformal threshold.
        """
        covered = score <= threshold
        self._coverage_history.append(covered)
        self._alpha_t = adaptive_conformal_update(
            self._alpha_t, covered, self.alpha_target, self.gamma
        )
        self._alpha_history.append(self._alpha_t)

    def get_threshold(self, scores_history: np.ndarray) -> float:
        """Compute the current adaptive threshold.

        Uses the current alpha_t (which may differ from the nominal alpha)
        to compute the conformal threshold on the available score history.

        Parameters
        ----------
        scores_history : np.ndarray
            All calibration scores seen so far.

        Returns
        -------
        float
            Adaptive conformal threshold.
        """
        # Clip alpha_t to avoid degenerate quantile levels
        alpha_eff = np.clip(self._alpha_t, 0.01, 0.99)
        return split_conformal_threshold(scores_history, alpha_eff)

    def coverage_trajectory(self) -> np.ndarray:
        """Rolling cumulative coverage over time.

        Returns
        -------
        np.ndarray
            Array of length len(coverage_history) with running average
            coverage up to each time step.
        """
        if not self._coverage_history:
            return np.array([])
        covered = np.array(self._coverage_history, dtype=float)
        return np.cumsum(covered) / np.arange(1, len(covered) + 1)

    @property
    def alpha_history(self) -> np.ndarray:
        """History of alpha_t values."""
        return np.array(self._alpha_history)


def generate_regime_data(
    n_calm: int = 500,
    n_crisis: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic return data with a volatility regime change.

    First ``n_calm`` observations are drawn from N(0, sigma_calm^2),
    followed by ``n_crisis`` observations from N(0, sigma_crisis^2).

    Parameters
    ----------
    n_calm : int
        Number of observations in the calm regime.
    n_crisis : int
        Number of observations in the crisis regime.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Combined return series of length n_calm + n_crisis.
    """
    rng = np.random.default_rng(seed)
    sigma_calm = 0.01
    sigma_crisis = 0.04
    calm = rng.normal(0, sigma_calm, n_calm)
    crisis = rng.normal(0, sigma_crisis, n_crisis)
    return np.concatenate([calm, crisis])


def run_aci_experiment(
    data: np.ndarray,
    model_fn: callable,
    alpha: float = 0.1,
    gamma: float = 0.01,
    cal_size: int = 100,
    window_size: int | None = None,
) -> dict:
    """Run a full ACI experiment on time-series data.

    Splits data into an initial calibration window and a test period.
    At each test step, computes the conformal threshold using the
    adaptive alpha_t on a sliding window of recent scores, evaluates
    coverage, and updates.

    Parameters
    ----------
    data : np.ndarray
        Full time series (1-D).
    model_fn : callable
        Function mapping (train_data,) -> point_predictions for the next
        step. For simplicity, uses the rolling mean.
    alpha : float
        Target miscoverage level.
    gamma : float
        ACI step size.
    cal_size : int
        Number of initial observations used for calibration.
    window_size : int or None
        Sliding window size for score history. If None, defaults to
        2 * cal_size to maintain adaptiveness to regime changes.

    Returns
    -------
    dict
        Keys: 'coverage_trajectory', 'thresholds', 'predictions',
        'alpha_history', 'scores'.
    """
    if window_size is None:
        window_size = 2 * cal_size

    aci = AdaptiveConformalInference(alpha=alpha, gamma=gamma)

    # Initial calibration: compute residuals on the calibration window
    cal_data = data[:cal_size]
    cal_preds = model_fn(cal_data)
    cal_scores = np.abs(cal_data[1:] - cal_preds[:-1])  # one-step-ahead residuals

    thresholds: list[float] = []
    predictions: list[float] = []
    scores_list: list[float] = []
    all_scores = list(cal_scores)

    # Online prediction phase
    for t in range(cal_size, len(data)):
        # Predict using history up to t
        history = data[:t]
        pred = model_fn(history)[-1]
        predictions.append(pred)

        # Adaptive threshold using sliding window of recent scores
        recent_scores = np.array(all_scores[-window_size:])
        threshold = aci.get_threshold(recent_scores)
        thresholds.append(threshold)

        # Observe true value and compute score
        true_val = data[t]
        score = abs(true_val - pred)
        scores_list.append(score)
        all_scores.append(score)

        # Update ACI
        aci.update(score, threshold)

    return {
        "coverage_trajectory": aci.coverage_trajectory(),
        "thresholds": np.array(thresholds),
        "predictions": np.array(predictions),
        "alpha_history": aci.alpha_history,
        "scores": np.array(scores_list),
    }
