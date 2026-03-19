"""Conformal prediction models for VaR and credit PD.

Implements:
    - QuantileRegressor: sklearn GradientBoosting wrapper for quantile loss
    - ConformalVaR: coverage-guaranteed VaR intervals via split conformal
    - ConformalPD: conformal prediction intervals for default probabilities

References:
    - Romano et al. (2019), Conformalized quantile regression.
    - Angelopoulos et al. (2024), Conformal risk control, JMLR 25(332).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from risk_analyst.models.conformal import (
    conformal_prediction_interval,
    cqr_interval,
    cqr_threshold,
    split_conformal_threshold,
)


class QuantileRegressor:
    """Quantile regression via sklearn GradientBoostingRegressor.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum depth of individual trees.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self._model: GradientBoostingRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, quantile: float) -> None:
        """Fit quantile regression using GradientBoosting with quantile loss.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Target array, shape (n_samples,).
        quantile : float
            Target quantile in (0, 1).
        """
        self._model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the fitted quantile.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted quantile values.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)


class ConformalVaR:
    """Conformal prediction intervals for Value-at-Risk.

    Uses Conformalized Quantile Regression (CQR) to produce asymmetric
    intervals with finite-sample coverage guarantees.

    Parameters
    ----------
    alpha : float
        Miscoverage level (e.g. 0.1 for 90% coverage).
    n_estimators : int
        Number of boosting stages for quantile regressors.
    max_depth : int
        Maximum tree depth.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 4,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.seed = seed
        self._lower_model = QuantileRegressor(n_estimators, max_depth, seed)
        self._upper_model = QuantileRegressor(n_estimators, max_depth, seed + 1)
        self._threshold: float | None = None

    def fit(
        self,
        returns_train: np.ndarray,
        returns_cal: np.ndarray,
    ) -> None:
        """Fit VaR model on training data and calibrate on calibration set.

        Constructs lagged features from the return series, fits lower and
        upper quantile regressors, then calibrates the CQR threshold.

        Parameters
        ----------
        returns_train : np.ndarray
            Training return series.
        returns_cal : np.ndarray
            Calibration return series.
        """
        X_train, y_train = self._make_features(returns_train)
        X_cal, y_cal = self._make_features(returns_cal)

        # Fit quantile regressors for lower and upper bounds
        self._lower_model.fit(X_train, y_train, quantile=self.alpha / 2)
        self._upper_model.fit(X_train, y_train, quantile=1 - self.alpha / 2)

        # Calibrate
        lower_cal = self._lower_model.predict(X_cal)
        upper_cal = self._upper_model.predict(X_cal)
        self._threshold = cqr_threshold(lower_cal, upper_cal, y_cal, self.alpha)

    def predict_interval(
        self, returns_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict conformal VaR intervals on test data.

        Parameters
        ----------
        returns_test : np.ndarray
            Test return series.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower, upper) interval bounds.
        """
        if self._threshold is None:
            raise RuntimeError("Model not calibrated. Call fit() first.")
        X_test, _ = self._make_features(returns_test)
        lower_pred = self._lower_model.predict(X_test)
        upper_pred = self._upper_model.predict(X_test)
        return cqr_interval(lower_pred, upper_pred, self._threshold)

    def coverage(
        self, returns_test: np.ndarray, true_losses: np.ndarray | None = None
    ) -> float:
        """Compute empirical coverage on test data.

        Parameters
        ----------
        returns_test : np.ndarray
            Test return series.
        true_losses : np.ndarray or None
            If provided, use these as ground truth. Otherwise, derive from
            returns_test using the same lagged feature construction.

        Returns
        -------
        float
            Empirical coverage rate.
        """
        _, y_test = self._make_features(returns_test)
        lower, upper = self.predict_interval(returns_test)
        if true_losses is not None:
            y_test = true_losses[: len(lower)]
        covered = (y_test >= lower) & (y_test <= upper)
        return float(np.mean(covered))

    @staticmethod
    def _make_features(
        returns: np.ndarray, n_lags: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create lagged feature matrix from return series.

        Parameters
        ----------
        returns : np.ndarray
            1-D return series of length T.
        n_lags : int
            Number of lags to use as features.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (X, y) where X has shape (T - n_lags, n_lags) and y has shape
            (T - n_lags,).
        """
        T = len(returns)
        X = np.column_stack(
            [returns[i : T - n_lags + i] for i in range(n_lags)]
        )
        y = returns[n_lags:]
        return X, y


class ConformalPD:
    """Conformal prediction intervals for probability of default.

    Uses split conformal prediction on a GradientBoosting classifier to
    produce coverage-guaranteed intervals for predicted default probabilities.

    Parameters
    ----------
    alpha : float
        Miscoverage level (e.g. 0.1 for 90% coverage).
    n_estimators : int
        Number of boosting stages for the classifier.
    max_depth : int
        Maximum tree depth.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 4,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.seed = seed
        self._classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        )
        self._threshold: float | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Train classifier and calibrate conformal scores.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels (0/1 default indicators).
        X_cal : np.ndarray
            Calibration features.
        y_cal : np.ndarray
            Calibration labels.
        """
        self._classifier.fit(X_train, y_train)

        # Nonconformity scores: |predicted_prob - true_label|
        probs_cal = self._classifier.predict_proba(X_cal)[:, 1]
        scores_cal = np.abs(probs_cal - y_cal)
        self._threshold = split_conformal_threshold(scores_cal, self.alpha)

    def predict_interval(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict conformal PD intervals.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (pd_lower, pd_upper) clipped to [0, 1].
        """
        if self._threshold is None:
            raise RuntimeError("Model not calibrated. Call fit() first.")
        probs_test = self._classifier.predict_proba(X_test)[:, 1]
        lower, upper = conformal_prediction_interval(probs_test, self._threshold)
        # Clip to valid probability range
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
        return lower, upper

    def coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute empirical coverage on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            True default indicators (0 or 1).

        Returns
        -------
        float
            Empirical coverage rate.
        """
        lower, upper = self.predict_interval(X_test)
        covered = (y_test >= lower) & (y_test <= upper)
        return float(np.mean(covered))
