"""Weight of Evidence (WoE) encoding and Information Value (IV) computation.

Implements classical scorecard feature engineering for credit risk modeling:
    1. WoE transformation: monotonic encoding capturing default predictiveness.
    2. IV screening: ranks features by predictive power (Siddiqi, 2017).

WoE for bin i:
    WoE_i = ln(Distribution of Goods_i / Distribution of Bads_i)

Information Value:
    IV = sum_i (Distr_Goods_i - Distr_Bads_i) * WoE_i

IV interpretation (Siddiqi, 2017, Table 3.2):
    < 0.02  : Not predictive
    0.02-0.1: Weak
    0.1-0.3 : Medium
    0.3-0.5 : Strong
    > 0.5   : Suspicious (overfitting or interaction)

References:
    - Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2nd ed., Ch. 3-4.
    - Thomas, L. C. (2009). *Consumer Credit Models*, Ch. 4: WoE scorecards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def woe_encode(
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    n_bins: int = 10,
) -> tuple[dict[int, float], float]:
    """Weight of Evidence encoding for a single feature.

    Bins the feature into ``n_bins`` quantile-based buckets and computes
    WoE per bin plus the total Information Value.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (1 = default / bad, 0 = non-default / good).
    feature : str
        Column name in *X* to encode.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    woe_dict : dict[int, float]
        Mapping from bin label (int) to WoE value.
    iv : float
        Total Information Value for this feature.
    """
    df = pd.DataFrame({"feature": X[feature].values, "target": y.values})

    # Quantile-based binning; duplicates='drop' handles ties
    df["bin"] = pd.qcut(df["feature"], q=n_bins, labels=False, duplicates="drop")

    total_goods = (df["target"] == 0).sum()
    total_bads = (df["target"] == 1).sum()

    # Guard against degenerate targets
    if total_goods == 0 or total_bads == 0:
        woe_dict = {b: 0.0 for b in df["bin"].unique()}
        return woe_dict, 0.0

    woe_dict: dict[int, float] = {}
    iv: float = 0.0

    for bin_label in sorted(df["bin"].dropna().unique()):
        mask = df["bin"] == bin_label
        n_goods = (df.loc[mask, "target"] == 0).sum()
        n_bads = (df.loc[mask, "target"] == 1).sum()

        # Laplace smoothing: add 0.5 to avoid log(0)
        dist_goods = (n_goods + 0.5) / (total_goods + 1.0)
        dist_bads = (n_bads + 0.5) / (total_bads + 1.0)

        woe_i = float(np.log(dist_goods / dist_bads))
        iv += (dist_goods - dist_bads) * woe_i

        woe_dict[int(bin_label)] = woe_i

    return woe_dict, float(iv)


def information_value(
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    n_bins: int = 10,
) -> float:
    """Compute Information Value for a single feature.

    Convenience wrapper around :func:`woe_encode` that returns only IV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (1 = default).
    feature : str
        Column name.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    float
        Information Value (non-negative).
    """
    _, iv = woe_encode(X, y, feature, n_bins=n_bins)
    return iv


def compute_all_iv(
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute IV for all numeric features, sorted descending.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (1 = default).
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    pd.DataFrame
        Two columns: ``feature`` and ``iv``, sorted by IV descending.
    """
    results: list[dict[str, object]] = []
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        iv = information_value(X, y, col, n_bins=n_bins)
        results.append({"feature": col, "iv": iv})

    df_iv = pd.DataFrame(results)
    df_iv = df_iv.sort_values("iv", ascending=False).reset_index(drop=True)
    return df_iv
