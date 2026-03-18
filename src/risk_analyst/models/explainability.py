"""Model explainability via SHAP and LIME.

Provides wrappers around SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro et al.,
2016) for both global and local post-hoc interpretability of credit risk models.

SHAP values satisfy the Shapley axioms from cooperative game theory:
    phi_j(f, x) = sum_{S subset N\\{j}} |S|!(|N|-|S|-1)!/|N|! * [f(S u {j}) - f(S)]

LIME fits a local linear model around a single prediction:
    xi = argmin_{g in G} L(f, g, pi_x) + Omega(g)

References:
    - Lundberg, S. M. & Lee, S.-I. (2017). NeurIPS. (SHAP)
    - Ribeiro, M. T. et al. (2016). KDD. (LIME)
    - SR 11-7 (2011): model interpretability requirements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def shap_summary(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """Compute SHAP values and return a summary DataFrame.

    Uses TreeExplainer for tree-based models and KernelExplainer as fallback.

    Parameters
    ----------
    model : fitted model
        A trained scikit-learn, XGBoost, or compatible model.
    X : pd.DataFrame or np.ndarray
        Feature matrix for which to compute SHAP values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature`` and ``mean_abs_shap``, sorted
        descending by mean absolute SHAP value (global importance).
    """
    import shap  # Lazy import -- heavy dependency

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fallback for non-tree models (e.g., logistic regression)
        if isinstance(X, pd.DataFrame):
            background = shap.sample(X, min(100, len(X)))
        else:
            background = X[:min(100, len(X))]
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X)

    # For binary classifiers, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # If 3D (n_samples, n_features, n_classes), take positive class
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.mean(np.abs(shap_values), axis=0)

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    summary_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return summary_df


def shap_waterfall(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    idx: int,
) -> dict[str, Any]:
    """SHAP waterfall data for a single prediction.

    Parameters
    ----------
    model : fitted model
        A trained model.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    idx : int
        Row index of the observation to explain.

    Returns
    -------
    dict
        Keys: ``base_value``, ``shap_values`` (1-D array), ``feature_values``
        (1-D array), ``feature_names`` (list[str]).
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        if isinstance(X, pd.DataFrame):
            background = shap.sample(X, min(100, len(X)))
        else:
            background = X[:min(100, len(X))]
        explainer = shap.KernelExplainer(model.predict_proba, background)

    if isinstance(X, pd.DataFrame):
        row = X.iloc[[idx]]
        feature_names = X.columns.tolist()
        feature_values = X.iloc[idx].values
    else:
        row = X[idx: idx + 1]
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_values = X[idx]

    shap_values = explainer.shap_values(row)

    # Handle binary classifier output
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[0, :, 1]
    else:
        shap_vals = shap_values[0]

    # Extract base value
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    return {
        "base_value": float(base_value),
        "shap_values": np.array(shap_vals, dtype=float),
        "feature_values": np.array(feature_values, dtype=float),
        "feature_names": feature_names,
    }


def lime_explain(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    idx: int,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """LIME explanation for a single prediction.

    Parameters
    ----------
    model : fitted model
        Must expose ``predict_proba`` method.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    idx : int
        Row index of the observation to explain.
    feature_names : list[str] or None
        Feature names. Inferred from DataFrame columns if *X* is a DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``weight`` (LIME coefficient for positive class).
    """
    import lime.lime_tabular  # Lazy import

    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_arr,
        feature_names=feature_names,
        class_names=["non-default", "default"],
        mode="classification",
        random_state=42,
    )

    explanation = explainer.explain_instance(
        X_arr[idx],
        model.predict_proba,
        num_features=len(feature_names),
    )

    lime_list = explanation.as_list()
    df = pd.DataFrame(lime_list, columns=["feature", "weight"])
    return df
