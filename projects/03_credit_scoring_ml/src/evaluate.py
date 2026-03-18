"""Evaluation metrics and diagnostic plots for credit scoring models.

Provides discriminatory-power metrics (AUC, KS, Gini), calibration
diagnostics, SHAP visualizations, and SR 11-7 model card generation.

Metrics implemented:
    - KS statistic: max |F_1(s) - F_0(s)| over score thresholds
    - Gini coefficient: 2*AUC - 1 (CAP curve area)
    - Brier score: mean squared calibration error

References:
    - Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2nd ed., Ch. 7-8.
    - SR 11-7 (2011): model risk management documentation.
    - Niculescu-Mizil & Caruana (2005): calibration of modern classifiers.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve


# ------------------------------------------------------------------
# Scalar metrics
# ------------------------------------------------------------------


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for a binary classifier.

    KS = max_s |F_1(s) - F_0(s)|

    where F_1 and F_0 are the cumulative distribution functions of the
    predicted probabilities for the positive and negative classes.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        KS statistic in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Sort by predicted probability
    sorted_idx = np.argsort(y_proba)
    y_sorted = y_true[sorted_idx]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg

    return float(np.max(np.abs(cum_pos - cum_neg)))


def compute_gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Gini coefficient: Gini = 2 * AUC - 1.

    The Gini coefficient measures the discriminatory power of the model.
    Equivalent to the area between the CAP curve and the diagonal.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.

    Returns
    -------
    float
        Gini coefficient in [-1, 1] (typically [0, 1] for useful models).
    """
    auc = roc_auc_score(y_true, y_proba)
    return 2.0 * auc - 1.0


# ------------------------------------------------------------------
# Diagnostic plots
# ------------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """ROC curve with AUC annotation.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.
    title : str
        Plot title.
    ax : matplotlib Axes or None
        Axes to plot on. Creates a new figure if None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_ks_chart(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """KS chart: cumulative distributions of defaults and non-defaults.

    The KS statistic is the maximum vertical distance between the two
    cumulative curves.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.
    ax : matplotlib Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    sorted_idx = np.argsort(y_proba)
    y_sorted = y_true[sorted_idx]
    proba_sorted = y_proba[sorted_idx]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    cum_pos = np.cumsum(y_sorted) / max(n_pos, 1)
    cum_neg = np.cumsum(1 - y_sorted) / max(n_neg, 1)

    ks = float(np.max(np.abs(cum_pos - cum_neg)))
    ks_idx = int(np.argmax(np.abs(cum_pos - cum_neg)))

    x_axis = np.arange(len(y_sorted)) / len(y_sorted)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.plot(x_axis, cum_pos, label="Defaults (cumulative)", linewidth=2)
    ax.plot(x_axis, cum_neg, label="Non-defaults (cumulative)", linewidth=2)
    ax.axvline(x=x_axis[ks_idx], color="red", linestyle="--", alpha=0.7)
    ax.annotate(
        f"KS = {ks:.4f}",
        xy=(x_axis[ks_idx], (cum_pos[ks_idx] + cum_neg[ks_idx]) / 2),
        fontsize=12,
        color="red",
        fontweight="bold",
    )
    ax.set_xlabel("Population fraction (sorted by score)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title(f"KS Chart (KS = {ks:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Reliability diagram (calibration curve).

    A well-calibrated model has predicted probabilities that match
    observed default rates in each bin.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.
    n_bins : int
        Number of calibration bins.
    ax : matplotlib Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.plot(
        mean_predicted_value, fraction_of_positives,
        "o-", linewidth=2, label="Model",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (observed)")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """SHAP beeswarm / bar summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix (n_samples x n_features).
    X : np.ndarray
        Feature values (same shape as shap_values).
    feature_names : list[str]
        Feature names.
    ax : matplotlib Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    else:
        fig = ax.get_figure()

    y_pos = np.arange(len(feature_names))
    ax.barh(
        y_pos,
        mean_abs[sorted_idx],
        color="#1f77b4",
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    return fig


# ------------------------------------------------------------------
# Model card (SR 11-7)
# ------------------------------------------------------------------


def model_card(
    model_name: str,
    metrics: dict[str, float],
    features: list[str],
    limitations: list[str],
) -> str:
    """Generate a markdown model card following SR 11-7 documentation standards.

    Produces a structured document covering model purpose, performance,
    features used, and known limitations -- as required by the Fed's
    model risk management guidance.

    Parameters
    ----------
    model_name : str
        Display name of the model (e.g., "XGBoost Credit Scorecard v1.0").
    metrics : dict[str, float]
        Performance metrics (e.g., ``{"auc": 0.82, "ks": 0.45}``).
    features : list[str]
        List of input features.
    limitations : list[str]
        Known limitations and assumptions.

    Returns
    -------
    str
        Markdown-formatted model card.
    """
    metrics_lines = "\n".join(
        f"| {name.upper()} | {value:.4f} |" for name, value in metrics.items()
    )
    features_lines = "\n".join(f"- {f}" for f in features)
    limitations_lines = "\n".join(f"- {l}" for l in limitations)

    card = f"""# Model Card: {model_name}

## 1. Model Overview

- **Model Name:** {model_name}
- **Model Type:** Probability of Default (PD) Scorecard
- **Intended Use:** Consumer credit risk assessment and decisioning
- **Developed By:** Risk Analyst Framework
- **Date:** Generated programmatically

## 2. Performance Metrics

| Metric | Value |
|--------|-------|
{metrics_lines}

## 3. Input Features

{features_lines}

## 4. Training Data

- Synthetic credit dataset with realistic feature distributions
- Default rate: ~5-10% (sub-prime consumer lending calibration)
- Temporal train/test split to avoid data leakage

## 5. Validation Approach

- Out-of-time (OOT) validation with temporal split
- Discriminatory power: AUC, KS statistic, Gini coefficient
- Calibration: Brier score, reliability diagram
- Explainability: SHAP global/local analysis

## 6. Known Limitations

{limitations_lines}

## 7. Regulatory Compliance

- Developed following SR 11-7 (Guidance on Model Risk Management) principles
- Model documentation includes: development rationale, assumptions,
  validation results, ongoing monitoring plan
- SHAP/LIME explanations available for adverse action notices (ECOA/Reg B)

## 8. Monitoring Plan

- Track population stability index (PSI) monthly
- Monitor KS and Gini on rolling 3-month cohorts
- Re-calibrate quarterly or when Brier score degrades >10%
- Champion-challenger framework for model updates
"""
    return card
