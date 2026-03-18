"""Markov regime-switching model for returns with switching variance.

Wraps ``statsmodels.tsa.regime_switching.markov_regression.MarkovRegression``
to fit a regime-switching model where both the mean and/or variance of
returns can differ across latent regimes.

Typical application: two-state model capturing "calm" (low-vol) and
"turbulent" (high-vol) market regimes.

References:
    - Hamilton (1989): seminal regime-switching econometrics paper.
    - Kim & Nelson (1999): state-space models with regime switching.
    - McNeil, Frey & Embrechts (2015), Ch. 4.6: regime-switching in risk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def fit_regime_switching(
    returns: np.ndarray | pd.Series,
    n_regimes: int = 2,
    switching_variance: bool = True,
) -> Any:
    """Fit a Markov regime-switching regression on returns.

    The model is:
        r_t = mu_{s_t} + sigma_{s_t} * epsilon_t,   epsilon_t ~ N(0,1)

    where s_t in {0, 1, ..., k-1} follows a first-order Markov chain
    with transition matrix P.

    Parameters
    ----------
    returns : array-like
        1-D return series (decimal form).
    n_regimes : int
        Number of hidden regimes (default 2).
    switching_variance : bool
        If ``True``, each regime has its own variance.

    Returns
    -------
    MarkovRegressionResults
        Fitted model containing regime parameters and smoothed
        probabilities.
    """
    y = np.asarray(returns, dtype=np.float64)
    if isinstance(returns, pd.Series):
        y_series = returns.copy()
    else:
        y_series = pd.Series(y)

    model = MarkovRegression(
        y_series,
        k_regimes=n_regimes,
        switching_variance=switching_variance,
    )
    result = model.fit(disp=False)
    return result


def regime_probabilities(fitted_model: Any) -> pd.DataFrame:
    """Extract smoothed regime probabilities from a fitted model.

    Uses the Kim (1994) smoothing algorithm (computed by statsmodels)
    to produce P(s_t = k | Y_T) for each regime k and time t.

    Parameters
    ----------
    fitted_model : MarkovRegressionResults
        Fitted regime-switching model.

    Returns
    -------
    pd.DataFrame
        T x k DataFrame of smoothed regime probabilities.
    """
    probs = fitted_model.smoothed_marginal_probabilities
    if isinstance(probs, pd.DataFrame):
        probs.columns = [f"regime_{i}" for i in range(probs.shape[1])]
        return probs

    # Fallback: ndarray
    n_regimes = probs.shape[1] if probs.ndim > 1 else 1
    cols = [f"regime_{i}" for i in range(n_regimes)]
    return pd.DataFrame(probs, columns=cols)


def regime_summary(fitted_model: Any) -> dict[str, Any]:
    """Summarize regime parameters: means, variances, transition matrix.

    Parameters
    ----------
    fitted_model : MarkovRegressionResults
        Fitted regime-switching model.

    Returns
    -------
    dict
        Keys:
        - ``"means"``: list of regime means.
        - ``"variances"``: list of regime variances.
        - ``"transition_matrix"``: k x k numpy array P_{ij} = P(s_{t+1}=j | s_t=i).
    """
    params = fitted_model.params
    n_regimes = fitted_model.k_regimes

    # Extract means
    means: list[float] = []
    for k in range(n_regimes):
        key = f"const[{k}]"
        if key in params.index:
            means.append(float(params[key]))
        else:
            means.append(0.0)

    # Extract variances (sigma^2)
    variances: list[float] = []
    for k in range(n_regimes):
        key = f"sigma2[{k}]"
        if key in params.index:
            variances.append(float(params[key]))

    if not variances:
        # Non-switching variance -- single sigma2
        if "sigma2" in params.index:
            shared_var = float(params["sigma2"])
            variances = [shared_var] * n_regimes

    # Transition matrix
    transition_matrix = np.array(fitted_model.regime_transition)

    return {
        "means": means,
        "variances": variances,
        "transition_matrix": transition_matrix,
    }
