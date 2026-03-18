"""Credit scoring model: logistic regression and XGBoost with WoE features.

Implements the ``CreditScoreModel`` class that encapsulates the full modeling
pipeline: WoE feature engineering, model fitting (logistic + XGBoost),
probability calibration, and SHAP-based explainability.

The logistic regression follows classical scorecard methodology (Siddiqi, 2017):
    1. Bin features into quantile buckets.
    2. Compute WoE per bin.
    3. Replace raw features with WoE values.
    4. Fit logistic regression on WoE-transformed features.

XGBoost operates on raw features (no WoE needed) with hyperparameters
externalized to a YAML config file.

References:
    - Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2nd ed., Ch. 5-6.
    - Chen, T. & Guestrin, C. (2016). XGBoost. KDD.
    - SR 11-7 (2011): model risk management requirements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from risk_analyst.models.credit import compute_all_iv, woe_encode
from risk_analyst.models.explainability import shap_summary, shap_waterfall


class CreditScoreModel:
    """End-to-end credit scoring model with WoE + logistic and XGBoost.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically loaded from default.yaml).
        Expected keys: ``features``, ``logistic``, ``xgboost``,
        ``calibration``, ``random_seed``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed: int = config.get("random_seed", 42)

        # Model references (set after fitting)
        self._logistic: LogisticRegression | None = None
        self._xgboost: XGBClassifier | None = None
        self._active_model: str = "xgboost"  # default prediction model
        self._calibrated_model: CalibratedClassifierCV | None = None

        # WoE encoding artifacts
        self._woe_maps: dict[str, dict[int, float]] = {}
        self._woe_bins: int = config.get("features", {}).get("woe_bins", 10)
        self._selected_features: list[str] = []
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # WoE feature engineering (for logistic)
    # ------------------------------------------------------------------

    def _fit_woe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Fit WoE maps and transform features.

        Parameters
        ----------
        X : pd.DataFrame
            Raw numeric features.
        y : pd.Series
            Binary target.

        Returns
        -------
        pd.DataFrame
            WoE-transformed feature matrix.
        """
        min_iv = self.config.get("features", {}).get("min_iv", 0.02)

        # Compute IV for feature selection
        iv_df = compute_all_iv(X, y, n_bins=self._woe_bins)
        self._selected_features = iv_df.loc[iv_df["iv"] >= min_iv, "feature"].tolist()

        # If no features pass IV threshold, keep top 5
        if len(self._selected_features) == 0:
            self._selected_features = iv_df["feature"].head(5).tolist()

        # Fit WoE maps
        self._woe_maps = {}
        for feat in self._selected_features:
            woe_dict, _ = woe_encode(X, y, feat, n_bins=self._woe_bins)
            self._woe_maps[feat] = woe_dict

        return self._transform_woe(X)

    def _transform_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted WoE maps to transform features.

        Parameters
        ----------
        X : pd.DataFrame
            Raw numeric features.

        Returns
        -------
        pd.DataFrame
            WoE-encoded features (only selected features).
        """
        result = pd.DataFrame(index=X.index)
        for feat in self._selected_features:
            woe_dict = self._woe_maps[feat]

            # Bin using the same number of quantile bins
            bins = pd.qcut(
                X[feat], q=self._woe_bins, labels=False, duplicates="drop",
            )

            # Map bin labels to WoE values; fill unknown bins with 0
            result[feat] = bins.map(woe_dict).fillna(0.0)

        return result

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit_logistic(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> LogisticRegression:
        """Fit logistic regression with WoE-encoded features.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (raw, before WoE).
        y_train : pd.Series
            Binary target.

        Returns
        -------
        LogisticRegression
            Fitted model.
        """
        X_woe = self._fit_woe(X_train, y_train)

        lr_config = self.config.get("logistic", {})
        self._logistic = LogisticRegression(
            C=lr_config.get("C", 1.0),
            max_iter=lr_config.get("max_iter", 1000),
            random_state=self.seed,
            solver="lbfgs",
        )
        self._logistic.fit(X_woe, y_train)
        self._active_model = "logistic"
        self._feature_names = list(X_woe.columns)
        return self._logistic

    def fit_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> XGBClassifier:
        """Fit XGBoost classifier on raw features.

        Auto-computes ``scale_pos_weight`` from the class ratio if not
        explicitly set in the config.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (numeric).
        y_train : pd.Series
            Binary target.

        Returns
        -------
        XGBClassifier
            Fitted model.
        """
        xgb_config = self.config.get("xgboost", {})

        # Auto-compute class weight if null
        scale_pos_weight = xgb_config.get("scale_pos_weight")
        if scale_pos_weight is None:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            scale_pos_weight = n_neg / max(n_pos, 1)

        self._xgboost = XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 200),
            max_depth=xgb_config.get("max_depth", 5),
            learning_rate=xgb_config.get("learning_rate", 0.1),
            min_child_weight=xgb_config.get("min_child_weight", 5),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            eval_metric=xgb_config.get("eval_metric", "auc"),
            random_state=self.seed,
            use_label_encoder=False,
            verbosity=0,
        )
        self._xgboost.fit(X_train, y_train)
        self._active_model = "xgboost"
        self._feature_names = list(X_train.columns)
        return self._xgboost

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability-of-default estimates.

        Uses the calibrated model if available, otherwise the active model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            1-D array of PD estimates (probability of class 1).
        """
        if self._calibrated_model is not None:
            X_input = self._prepare_features(X)
            return self._calibrated_model.predict_proba(X_input)[:, 1]

        if self._active_model == "logistic" and self._logistic is not None:
            X_woe = self._transform_woe(X)
            return self._logistic.predict_proba(X_woe)[:, 1]

        if self._active_model == "xgboost" and self._xgboost is not None:
            return self._xgboost.predict_proba(X)[:, 1]

        raise RuntimeError("No model has been fitted. Call fit_logistic or fit_xgboost first.")

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the active model."""
        if self._active_model == "logistic":
            return self._transform_woe(X)
        return X

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model discriminatory power and calibration.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            True labels.

        Returns
        -------
        dict[str, float]
            Keys: ``auc``, ``ks``, ``gini``, ``brier``.
        """
        y_proba = self.predict_proba(X_test)

        auc = roc_auc_score(y_test, y_proba)
        gini = 2.0 * auc - 1.0

        # KS statistic: max separation between cumulative distributions
        y_true_arr = np.asarray(y_test)
        sorted_idx = np.argsort(y_proba)
        y_sorted = y_true_arr[sorted_idx]
        n_pos = y_sorted.sum()
        n_neg = len(y_sorted) - n_pos
        cum_pos = np.cumsum(y_sorted) / max(n_pos, 1)
        cum_neg = np.cumsum(1 - y_sorted) / max(n_neg, 1)
        ks = float(np.max(np.abs(cum_pos - cum_neg)))

        brier = brier_score_loss(y_test, y_proba)

        return {
            "auc": float(auc),
            "ks": float(ks),
            "gini": float(gini),
            "brier": float(brier),
        }

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain(
        self,
        X: pd.DataFrame,
        idx: int | None = None,
    ) -> pd.DataFrame | dict[str, Any]:
        """SHAP-based explanations.

        If ``idx`` is None, returns global feature importance (summary).
        If ``idx`` is an integer, returns local waterfall data for that row.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        idx : int or None
            Row index for local explanation.

        Returns
        -------
        pd.DataFrame or dict
            Global summary DataFrame or local waterfall dict.
        """
        model = self._get_active_model()
        X_input = self._prepare_features(X)

        if idx is None:
            return shap_summary(model, X_input)
        return shap_waterfall(model, X_input, idx)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        method: str = "isotonic",
    ) -> CalibratedClassifierCV:
        """Calibrate predicted probabilities.

        Wraps the active model with scikit-learn's CalibratedClassifierCV
        using the provided validation set.

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation features.
        y_val : pd.Series
            Validation labels.
        method : str
            ``"isotonic"`` (non-parametric) or ``"sigmoid"`` (Platt scaling).

        Returns
        -------
        CalibratedClassifierCV
            Calibrated model.
        """
        model = self._get_active_model()
        X_input = self._prepare_features(X_val)

        # FrozenEstimator prevents re-fitting the base model (sklearn >= 1.8)
        self._calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
            cv=2,
        )
        self._calibrated_model.fit(X_input, y_val)
        return self._calibrated_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_model(self) -> LogisticRegression | XGBClassifier:
        """Return the currently active fitted model."""
        if self._active_model == "logistic" and self._logistic is not None:
            return self._logistic
        if self._active_model == "xgboost" and self._xgboost is not None:
            return self._xgboost
        raise RuntimeError("No model fitted.")
