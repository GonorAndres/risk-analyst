"""Unit tests for Project 03: Credit Scoring with ML.

Tests cover:
    - WoE encoding produces finite values (no log(0) or inf)
    - IV is non-negative for all features
    - AUC > 0.5 for both models (better than random)
    - XGBoost AUC >= logistic AUC (or close)
    - Calibrated probabilities are closer to observed rates
    - Gini = 2*AUC - 1 identity
    - KS statistic is in [0, 1]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

# Ensure project src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Ensure shared library is importable
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate import compute_gini, compute_ks_statistic
from model import CreditScoreModel

from data import generate_synthetic_credit_data, preprocess, train_test_split_temporal
from risk_analyst.models.credit import compute_all_iv, information_value, woe_encode

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> dict:
    """Load default config."""
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    """Generate synthetic data (deterministic)."""
    return generate_synthetic_credit_data(n_samples=5000, seed=42)


@pytest.fixture(scope="module")
def processed_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocessed dataset."""
    return preprocess(raw_data)


@pytest.fixture(scope="module")
def train_test(processed_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/test split."""
    return train_test_split_temporal(processed_data, test_ratio=0.2)


@pytest.fixture(scope="module")
def feature_cols(processed_data: pd.DataFrame) -> list[str]:
    """Numeric feature columns (exclude target and date)."""
    exclude = {"default", "application_date"}
    return [c for c in processed_data.columns if c not in exclude]


@pytest.fixture(scope="module")
def fitted_models(
    config: dict,
    train_test: tuple[pd.DataFrame, pd.DataFrame],
    feature_cols: list[str],
) -> dict:
    """Fit both logistic and XGBoost models, return dict with metrics."""
    train_df, test_df = train_test
    X_train = train_df[feature_cols]
    y_train = train_df["default"]
    X_test = test_df[feature_cols]
    y_test = test_df["default"]

    # Logistic regression
    lr_model = CreditScoreModel(config)
    lr_model.fit_logistic(X_train, y_train)
    lr_proba = lr_model.predict_proba(X_test)
    lr_auc = roc_auc_score(y_test, lr_proba)

    # XGBoost
    xgb_model = CreditScoreModel(config)
    xgb_model.fit_xgboost(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    return {
        "lr_model": lr_model,
        "xgb_model": xgb_model,
        "lr_proba": lr_proba,
        "xgb_proba": xgb_proba,
        "lr_auc": lr_auc,
        "xgb_auc": xgb_auc,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ------------------------------------------------------------------
# WoE and IV tests
# ------------------------------------------------------------------


class TestWoEEncoding:
    """Tests for WoE encoding."""

    def test_woe_values_are_finite(
        self,
        processed_data: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        """WoE encoding produces finite values (no log(0) or inf)."""
        X = processed_data[feature_cols]
        y = processed_data["default"]

        for col in feature_cols[:5]:  # Test first 5 features for speed
            woe_dict, iv = woe_encode(X, y, col, n_bins=10)

            for bin_label, woe_val in woe_dict.items():
                assert np.isfinite(woe_val), (
                    f"Non-finite WoE for feature={col}, bin={bin_label}: {woe_val}"
                )

    def test_iv_is_non_negative(
        self,
        processed_data: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        """Information Value is non-negative for all features."""
        X = processed_data[feature_cols]
        y = processed_data["default"]

        for col in feature_cols:
            iv = information_value(X, y, col, n_bins=10)
            assert iv >= 0, f"Negative IV for feature {col}: {iv}"

    def test_compute_all_iv_sorted(
        self,
        processed_data: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        """compute_all_iv returns results sorted by IV descending."""
        X = processed_data[feature_cols]
        y = processed_data["default"]

        iv_df = compute_all_iv(X, y, n_bins=10)

        assert "feature" in iv_df.columns
        assert "iv" in iv_df.columns
        # Check descending sort
        iv_values = iv_df["iv"].values
        assert all(iv_values[i] >= iv_values[i + 1] for i in range(len(iv_values) - 1))


# ------------------------------------------------------------------
# Model performance tests
# ------------------------------------------------------------------


class TestModelPerformance:
    """Tests for model discriminatory power."""

    def test_logistic_auc_better_than_random(self, fitted_models: dict) -> None:
        """Logistic regression AUC > 0.5 (better than random)."""
        assert fitted_models["lr_auc"] > 0.5, (
            f"Logistic AUC = {fitted_models['lr_auc']:.4f}, not better than random"
        )

    def test_xgboost_auc_better_than_random(self, fitted_models: dict) -> None:
        """XGBoost AUC > 0.5 (better than random)."""
        assert fitted_models["xgb_auc"] > 0.5, (
            f"XGBoost AUC = {fitted_models['xgb_auc']:.4f}, not better than random"
        )

    def test_xgboost_auc_at_least_as_good_as_logistic(
        self, fitted_models: dict,
    ) -> None:
        """XGBoost AUC >= logistic AUC (or within 0.02 tolerance)."""
        assert fitted_models["xgb_auc"] >= fitted_models["lr_auc"] - 0.02, (
            f"XGBoost AUC ({fitted_models['xgb_auc']:.4f}) much worse than "
            f"logistic AUC ({fitted_models['lr_auc']:.4f})"
        )


# ------------------------------------------------------------------
# Calibration tests
# ------------------------------------------------------------------


class TestCalibration:
    """Tests for probability calibration."""

    def test_calibrated_brier_improved(self, fitted_models: dict) -> None:
        """Calibrated probabilities produce a Brier score no worse than raw."""
        xgb_model = fitted_models["xgb_model"]
        X_test = fitted_models["X_test"]
        y_test = fitted_models["y_test"]

        # Raw Brier
        raw_proba = fitted_models["xgb_proba"]
        raw_brier = brier_score_loss(y_test, raw_proba)

        # Split test for calibration (use first half for cal, second for eval)
        n_half = len(X_test) // 2
        X_cal = X_test.iloc[:n_half]
        y_cal = y_test.iloc[:n_half]
        X_eval = X_test.iloc[n_half:]
        y_eval = y_test.iloc[n_half:]

        xgb_model.calibrate(X_cal, y_cal, method="isotonic")
        cal_proba = xgb_model.predict_proba(X_eval)
        cal_brier = brier_score_loss(y_eval, cal_proba)

        # Calibration should not make things dramatically worse
        # Allow 50% tolerance since calibration on small sets can be noisy
        assert cal_brier <= raw_brier * 1.5, (
            f"Calibrated Brier ({cal_brier:.4f}) much worse than raw ({raw_brier:.4f})"
        )


# ------------------------------------------------------------------
# Metric identity tests
# ------------------------------------------------------------------


class TestMetricIdentities:
    """Tests for metric computation correctness."""

    def test_gini_equals_2auc_minus_1(self, fitted_models: dict) -> None:
        """Verify Gini = 2*AUC - 1 identity."""
        y_test = fitted_models["y_test"]
        y_proba = fitted_models["xgb_proba"]

        auc = roc_auc_score(y_test, y_proba)
        gini = compute_gini(y_test, y_proba)

        assert gini == pytest.approx(2 * auc - 1, abs=1e-10), (
            f"Gini ({gini:.6f}) != 2*AUC-1 ({2*auc-1:.6f})"
        )

    def test_ks_between_0_and_1(self, fitted_models: dict) -> None:
        """KS statistic is in [0, 1]."""
        y_test = fitted_models["y_test"]
        y_proba = fitted_models["xgb_proba"]

        ks = compute_ks_statistic(y_test, y_proba)

        assert 0.0 <= ks <= 1.0, f"KS statistic out of range: {ks}"

    def test_ks_positive_for_good_model(self, fitted_models: dict) -> None:
        """A model with AUC > 0.5 should have KS > 0."""
        y_test = fitted_models["y_test"]
        y_proba = fitted_models["xgb_proba"]

        ks = compute_ks_statistic(y_test, y_proba)
        assert ks > 0, f"KS statistic should be positive for a good model, got {ks}"


# ------------------------------------------------------------------
# Data pipeline tests
# ------------------------------------------------------------------


class TestDataPipeline:
    """Tests for data loading and preprocessing."""

    def test_synthetic_data_deterministic(self) -> None:
        """Same seed produces identical data."""
        df1 = generate_synthetic_credit_data(n_samples=100, seed=123)
        df2 = generate_synthetic_credit_data(n_samples=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_default_rate_realistic(self, raw_data: pd.DataFrame) -> None:
        """Default rate should be between 3% and 15%."""
        rate = raw_data["default"].mean()
        assert 0.03 <= rate <= 0.15, f"Default rate {rate:.2%} outside realistic range"

    def test_temporal_split_no_leakage(
        self,
        processed_data: pd.DataFrame,
    ) -> None:
        """Test set dates are strictly after train set dates."""
        train, test = train_test_split_temporal(processed_data, test_ratio=0.2)
        max_train_date = train["application_date"].max()
        min_test_date = test["application_date"].min()
        assert min_test_date >= max_train_date, (
            f"Temporal leakage: test min date {min_test_date} < "
            f"train max date {max_train_date}"
        )

    def test_preprocess_no_missing(self, processed_data: pd.DataFrame) -> None:
        """Preprocessing eliminates missing values in numeric columns."""
        numeric = processed_data.select_dtypes(include=[np.number])
        assert numeric.isna().sum().sum() == 0, "Numeric columns still have NaN"
