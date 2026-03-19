"""Project 04 -- Diagnostics and visualization for volatility models.

Functions for visual assessment and statistical testing of GARCH-family
and regime-switching models.

Plots follow the project convention of returning ``matplotlib.figure.Figure``
objects so callers can save or display as needed.

References:
    - Ljung & Box (1978): portmanteau test for autocorrelation.
    - Engle & Ng (1993): news impact curves and standardized residual diagnostics.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelResult
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------------------------------------------------------------
# Time-series overlays
# ---------------------------------------------------------------------------


def plot_conditional_volatility(
    returns: np.ndarray | pd.Series,
    fitted_model: ARCHModelResult,
    title: str = "Returns with Conditional Volatility",
) -> plt.Figure:
    """Plot returns with the conditional sigma +/- band.

    Parameters
    ----------
    returns : array-like
        Decimal return series.
    fitted_model : ARCHModelResult
        Fitted GARCH-family model.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Conditional volatility from the fitted model (pct -> decimal)
    cond_vol_pct = fitted_model.conditional_volatility
    cond_vol = cond_vol_pct / 100.0

    if isinstance(returns, pd.Series):
        idx = returns.index
        ret_vals = returns.values
    else:
        idx = np.arange(len(returns))
        ret_vals = np.asarray(returns)

    # Align lengths (model may have dropped initial observations)
    n = min(len(ret_vals), len(cond_vol))
    idx = idx[-n:]
    ret_vals = ret_vals[-n:]
    sigma = np.asarray(cond_vol)[-n:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, ret_vals, color="steelblue", linewidth=0.5, alpha=0.7, label="Returns")
    ax.plot(idx, sigma, color="red", linewidth=1.0, label="+sigma")
    ax.plot(idx, -sigma, color="red", linewidth=1.0, label="-sigma")
    ax.fill_between(idx, -sigma, sigma, color="red", alpha=0.08)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


def plot_standardized_residuals(
    fitted_model: ARCHModelResult,
) -> plt.Figure:
    """Histogram and QQ plot of standardized residuals.

    Standardized residuals z_t = epsilon_t / sigma_t should be
    approximately i.i.d. if the GARCH model is well-specified.

    Parameters
    ----------
    fitted_model : ARCHModelResult

    Returns
    -------
    matplotlib.figure.Figure
    """
    std_resid = fitted_model.std_resid.dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax_hist = axes[0]
    ax_hist.hist(std_resid, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    x_grid = np.linspace(std_resid.min(), std_resid.max(), 300)
    ax_hist.plot(x_grid, stats.norm.pdf(x_grid), "r-", lw=1.5, label="N(0,1)")
    ax_hist.set_title("Standardized Residuals -- Histogram")
    ax_hist.set_xlabel("z_t")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()

    # QQ plot
    ax_qq = axes[1]
    stats.probplot(std_resid, dist="norm", plot=ax_qq)
    ax_qq.set_title("Standardized Residuals -- QQ Plot")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regime-switching visualization
# ---------------------------------------------------------------------------


def plot_regime_probabilities(
    returns: np.ndarray | pd.Series,
    regime_model: Any,
) -> plt.Figure:
    """Returns time series with regime probability shading.

    The background is shaded according to the smoothed probability
    of being in the high-volatility regime.

    Parameters
    ----------
    returns : array-like
        Decimal return series.
    regime_model : MarkovRegressionResults
        Fitted regime-switching model.

    Returns
    -------
    matplotlib.figure.Figure
    """
    probs = regime_model.smoothed_marginal_probabilities

    if isinstance(returns, pd.Series):
        idx = returns.index
        ret_vals = returns.values
    else:
        idx = np.arange(len(returns))
        ret_vals = np.asarray(returns)

    # Identify the high-vol regime by comparing regime variances
    params = regime_model.params
    n_regimes = regime_model.k_regimes
    regime_vars: list[float] = []
    for k in range(n_regimes):
        key = f"sigma2[{k}]"
        if key in params.index:
            regime_vars.append(float(params[key]))
        else:
            regime_vars.append(0.0)

    high_vol_regime = int(np.argmax(regime_vars)) if regime_vars else 1

    # Align lengths
    n = min(len(ret_vals), probs.shape[0])
    idx = idx[-n:]
    ret_vals = ret_vals[-n:]

    if isinstance(probs, pd.DataFrame):
        high_prob = probs.iloc[-n:, high_vol_regime].values
    else:
        high_prob = probs[-n:, high_vol_regime]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Returns
    ax1.plot(idx, ret_vals, color="steelblue", linewidth=0.5, alpha=0.8)
    ax1.fill_between(
        idx, ret_vals.min(), ret_vals.max(),
        where=high_prob > 0.5,
        color="salmon", alpha=0.25, label="High-vol regime",
    )
    ax1.set_ylabel("Return")
    ax1.set_title("Returns with Regime Probability Shading")
    ax1.legend(loc="upper right")

    # Regime probability
    ax2.fill_between(idx, 0, high_prob, color="salmon", alpha=0.5)
    ax2.set_ylabel(f"P(regime {high_vol_regime})")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1)
    ax2.set_title("Smoothed High-Volatility Regime Probability")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model comparison chart
# ---------------------------------------------------------------------------


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing AIC and BIC across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of ``VolatilityModel.compare_models()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["AIC", "BIC"]):
        vals = comparison_df[metric]
        colors = ["forestgreen" if v == vals.min() else "steelblue" for v in vals]
        ax.barh(comparison_df.index, vals, color=colors, edgecolor="white")
        ax.set_xlabel(metric)
        ax.set_title(f"Model Comparison -- {metric}")
        ax.invert_yaxis()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Volatility term structure
# ---------------------------------------------------------------------------


def plot_volatility_term_structure(
    fitted_model: ARCHModelResult,
    horizons: list[int] | None = None,
) -> plt.Figure:
    """Multi-step ahead volatility forecasts forming a term structure.

    Parameters
    ----------
    fitted_model : ARCHModelResult
        Fitted model.
    horizons : list[int] | None
        Horizons to plot (default: 1 through 20).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if horizons is None:
        horizons = list(range(1, 21))

    max_h = max(horizons)
    fcast = fitted_model.forecast(horizon=max_h)
    variance_pct = fcast.variance.dropna().iloc[-1]
    sigma_decimal = np.sqrt(variance_pct.values[:max_h]) / 100.0

    # Select only the requested horizons (1-indexed in horizons list)
    sigma_at_h = [sigma_decimal[h - 1] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(horizons, sigma_at_h, "o-", color="steelblue", linewidth=1.5)
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Conditional Sigma (decimal)")
    ax.set_title("Volatility Term Structure")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def ljung_box_test(
    residuals: np.ndarray | pd.Series,
    lags: int = 10,
) -> pd.DataFrame:
    """Ljung-Box test on squared standardized residuals.

    Under a well-specified GARCH model the squared standardized
    residuals z_t^2 should be serially uncorrelated.  Rejecting H0
    (p-value < 0.05) indicates remaining volatility clustering.

    Parameters
    ----------
    residuals : array-like
        Standardized residuals z_t (not squared -- squaring is done here).
    lags : int
        Maximum lag for the test.

    Returns
    -------
    pd.DataFrame
        Ljung-Box test statistics and p-values per lag.
    """
    z2 = np.asarray(residuals, dtype=np.float64) ** 2
    result = acorr_ljungbox(z2, lags=lags, return_df=True)
    return result
