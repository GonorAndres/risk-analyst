"""Copula dependency models: fitting, sampling, and tail dependence.

Implements elliptical (Gaussian, Student-t) and Archimedean (Clayton, Gumbel,
Frank) copulas using only scipy and numpy -- no external copula libraries.

Estimation approaches:
    - Elliptical: rank correlation + profile MLE for degrees of freedom.
    - Archimedean: Kendall's tau inversion (closed-form for Clayton/Gumbel)
      and numerical inversion for Frank.

References:
    - Nelsen (2006), *An Introduction to Copulas*, Ch. 4--5.
    - Joe (2014), *Dependence Modeling with Copulas*, Ch. 3.
    - McNeil, Frey & Embrechts (2015), Ch. 7: copula models.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Probability integral transform
# ---------------------------------------------------------------------------


def pit_transform(
    data: np.ndarray,
    method: str = "empirical",
) -> np.ndarray:
    """Probability integral transform (PIT).

    Transforms each column of *data* to approximately Uniform(0, 1).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n,) or (n, d).
    method : {"empirical", "parametric"}
        ``"empirical"`` uses rank/(n+1) (Weibull plotting position).
        ``"parametric"`` fits a normal distribution to each column and
        applies its CDF.

    Returns
    -------
    np.ndarray
        Uniform pseudo-observations with the same shape as *data*.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    n, d = data.shape
    u = np.empty_like(data)

    if method == "empirical":
        for j in range(d):
            ranks = stats.rankdata(data[:, j], method="ordinal")
            u[:, j] = ranks / (n + 1)
    elif method == "parametric":
        for j in range(d):
            mu = np.mean(data[:, j])
            sigma = np.std(data[:, j], ddof=1)
            u[:, j] = stats.norm.cdf(data[:, j], loc=mu, scale=sigma)
    else:
        raise ValueError(
            f"Unknown PIT method '{method}'. Choose 'empirical' or 'parametric'."
        )

    if squeeze:
        return u.ravel()
    return u


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------


def gaussian_copula_fit(u_data: np.ndarray) -> dict:
    """Fit a Gaussian copula by estimating the correlation matrix.

    Transforms uniform marginals to standard normals via the inverse CDF,
    then computes the sample correlation matrix.

    Parameters
    ----------
    u_data : np.ndarray
        Uniform pseudo-observations of shape (n, d), each column in (0, 1).

    Returns
    -------
    dict
        ``{"family": "gaussian", "corr_matrix": ndarray, "d": int}``
    """
    u_data = np.asarray(u_data, dtype=np.float64)
    # Clip to avoid +/-inf from the normal quantile function
    u_clipped = np.clip(u_data, 1e-10, 1 - 1e-10)
    z = stats.norm.ppf(u_clipped)
    corr_matrix = np.corrcoef(z, rowvar=False)
    return {
        "family": "gaussian",
        "corr_matrix": corr_matrix,
        "d": u_data.shape[1],
    }


# ---------------------------------------------------------------------------
# Student-t copula
# ---------------------------------------------------------------------------


def _t_copula_loglik(
    df: float,
    z: np.ndarray,
    corr_matrix: np.ndarray,
) -> float:
    """Negative log-likelihood of the t-copula (profile over df).

    Given normal-score data *z* (from uniform marginals), the t-copula
    density at observation x_i (d-dimensional) is:

        c(u) = f_{t,d}(t_df^{-1}(u_1), ..., t_df^{-1}(u_d); R, df)
               / prod_{j=1}^{d} f_{t,1}(t_df^{-1}(u_j); df)

    We maximise over df with R fixed at the rank correlation estimate.
    """
    n, d = z.shape
    # Transform normal scores to t-scores with candidate df
    t_scores = stats.t.ppf(stats.norm.cdf(z), df=df)

    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        return 1e12  # non-PD matrix -- penalise

    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # Multivariate t log-density (unnormalised, up to additive const)
    inv_corr = np.linalg.inv(corr_matrix)
    quad = np.sum((t_scores @ inv_corr) * t_scores, axis=1)

    # Joint log-density of multivariate t
    log_joint = (
        _log_mvt_const(d, df)
        - 0.5 * log_det
        - 0.5 * (df + d) * np.log(1.0 + quad / df)
    )

    # Marginal log-densities of univariate t
    log_marginals = np.sum(stats.t.logpdf(t_scores, df=df), axis=1)

    # Copula log-likelihood = sum of log(c(u_i))
    ll = np.sum(log_joint - log_marginals)
    return -ll  # negative for minimisation


def _log_mvt_const(d: int, df: float) -> float:
    """Log of the normalising constant for the d-variate t distribution."""
    from scipy.special import gammaln

    return (
        gammaln(0.5 * (df + d))
        - gammaln(0.5 * df)
        - 0.5 * d * np.log(df * np.pi)
    )


def t_copula_fit(
    u_data: np.ndarray,
    df_range: tuple[float, float] = (2, 30),
) -> dict:
    """Fit a Student-t copula by profile maximum likelihood.

    Estimates the correlation matrix from normal-score ranks, then
    optimises the degrees-of-freedom parameter by profile MLE.

    Parameters
    ----------
    u_data : np.ndarray
        Uniform pseudo-observations of shape (n, d).
    df_range : tuple
        Bounds for the degrees-of-freedom search.

    Returns
    -------
    dict
        ``{"family": "t", "corr_matrix": ndarray, "df": float, "d": int}``
    """
    u_data = np.asarray(u_data, dtype=np.float64)
    u_clipped = np.clip(u_data, 1e-10, 1 - 1e-10)
    z = stats.norm.ppf(u_clipped)
    corr_matrix = np.corrcoef(z, rowvar=False)

    # Profile MLE over df
    result = optimize.minimize_scalar(
        _t_copula_loglik,
        bounds=df_range,
        method="bounded",
        args=(z, corr_matrix),
    )
    df_hat = result.x

    return {
        "family": "t",
        "corr_matrix": corr_matrix,
        "df": float(df_hat),
        "d": u_data.shape[1],
    }


# ---------------------------------------------------------------------------
# Archimedean copulas (bivariate)
# ---------------------------------------------------------------------------


def _kendall_tau(u_data: np.ndarray) -> float:
    """Compute Kendall's tau for a bivariate sample."""
    if u_data.shape[1] != 2:
        raise ValueError("Kendall's tau inversion requires bivariate data.")
    tau, _ = stats.kendalltau(u_data[:, 0], u_data[:, 1])
    return float(tau)


def clayton_copula_fit(u_data: np.ndarray) -> dict:
    """Fit a Clayton copula (bivariate) via Kendall's tau inversion.

    theta = 2 * tau / (1 - tau),  theta > 0  =>  tau > 0.

    Parameters
    ----------
    u_data : np.ndarray
        Uniform pseudo-observations of shape (n, 2).

    Returns
    -------
    dict
        ``{"family": "clayton", "theta": float}``
    """
    tau = _kendall_tau(u_data)
    if tau <= 0:
        warnings.warn(
            f"Kendall's tau = {tau:.4f} <= 0; Clayton requires positive dependence. "
            "Setting theta to a small positive value.",
            stacklevel=2,
        )
        theta = 0.01
    else:
        theta = 2.0 * tau / (1.0 - tau)
    return {"family": "clayton", "theta": float(theta)}


def gumbel_copula_fit(u_data: np.ndarray) -> dict:
    """Fit a Gumbel copula (bivariate) via Kendall's tau inversion.

    theta = 1 / (1 - tau),  theta >= 1  =>  tau >= 0.

    Parameters
    ----------
    u_data : np.ndarray
        Uniform pseudo-observations of shape (n, 2).

    Returns
    -------
    dict
        ``{"family": "gumbel", "theta": float}``
    """
    tau = _kendall_tau(u_data)
    if tau <= 0:
        warnings.warn(
            f"Kendall's tau = {tau:.4f} <= 0; Gumbel requires positive dependence. "
            "Setting theta = 1 (independence).",
            stacklevel=2,
        )
        theta = 1.0
    else:
        theta = 1.0 / (1.0 - tau)
    return {"family": "gumbel", "theta": float(theta)}


def _frank_tau_equation(theta: float) -> float:
    """Kendall's tau for Frank copula: tau = 1 - 4/theta * (1 - D_1(theta))

    where D_1(theta) = (1/theta) * integral_0^theta t/(exp(t)-1) dt
    is the first Debye function.
    """
    if abs(theta) < 1e-10:
        return 0.0

    from scipy.integrate import quad

    def integrand(t: float) -> float:
        if abs(t) < 1e-15:
            return 1.0
        return t / (np.exp(t) - 1.0)

    debye, _ = quad(integrand, 0, abs(theta))
    debye /= abs(theta)
    tau = 1.0 - 4.0 / theta + 4.0 / (theta**2) * abs(theta) * debye
    # Simplified: tau = 1 - 4*(1 - D_1(theta)) / theta
    return tau


def frank_copula_fit(u_data: np.ndarray) -> dict:
    """Fit a Frank copula (bivariate) via Kendall's tau inversion.

    Numerically inverts the relationship tau = 1 - 4/theta + 4*D_1(theta)/theta.

    Parameters
    ----------
    u_data : np.ndarray
        Uniform pseudo-observations of shape (n, 2).

    Returns
    -------
    dict
        ``{"family": "frank", "theta": float}``
    """
    tau = _kendall_tau(u_data)

    if abs(tau) < 1e-10:
        return {"family": "frank", "theta": 0.0}

    # Numerical inversion: find theta such that frank_tau(theta) = tau
    def objective(theta: float) -> float:
        return _frank_tau_equation(theta) - tau

    # Frank allows theta in (-inf, 0) u (0, +inf); sign matches tau
    if tau > 0:
        bracket = (0.01, 100.0)
    else:
        bracket = (-100.0, -0.01)

    try:
        sol = optimize.brentq(objective, bracket[0], bracket[1])
    except ValueError:
        # Expand search range
        if tau > 0:
            sol = optimize.brentq(objective, 1e-4, 500.0)
        else:
            sol = optimize.brentq(objective, -500.0, -1e-4)

    return {"family": "frank", "theta": float(sol)}


# ---------------------------------------------------------------------------
# Copula sampling
# ---------------------------------------------------------------------------


def copula_sample(
    params: dict,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Draw samples from a fitted copula.

    Parameters
    ----------
    params : dict
        Fitted copula parameters (output of ``*_copula_fit`` functions).
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Uniform samples of shape (n_samples, d).
    """
    rng = np.random.default_rng(seed)
    family = params["family"]

    if family == "gaussian":
        return _sample_gaussian(params, n_samples, rng)
    elif family == "t":
        return _sample_t(params, n_samples, rng)
    elif family == "clayton":
        return _sample_clayton(params, n_samples, rng)
    elif family == "gumbel":
        return _sample_gumbel(params, n_samples, rng)
    elif family == "frank":
        return _sample_frank(params, n_samples, rng)
    else:
        raise ValueError(f"Unknown copula family '{family}'.")


def _sample_gaussian(
    params: dict, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample from Gaussian copula via Cholesky + Phi."""
    corr = params["corr_matrix"]
    d = corr.shape[0]
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_samples, d))
    y = z @ L.T
    return stats.norm.cdf(y)


def _sample_t(
    params: dict, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample from t-copula via Cholesky + chi2 mixing."""
    corr = params["corr_matrix"]
    df = params["df"]
    d = corr.shape[0]
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_samples, d))
    y = z @ L.T

    # chi-squared mixing variable
    s = rng.chisquare(df, size=n_samples)
    t_samples = y / np.sqrt(s / df)[:, np.newaxis]
    return stats.t.cdf(t_samples, df=df)


def _sample_clayton(
    params: dict, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample bivariate Clayton copula via conditional method.

    Algorithm (Nelsen, 2006, p. 41):
        1. u1 ~ Uniform(0, 1)
        2. t  ~ Uniform(0, 1)
        3. u2 = (u1^{-theta} * (t^{-theta/(1+theta)} - 1) + 1)^{-1/theta}
    """
    theta = params["theta"]
    u1 = rng.uniform(size=n_samples)
    t = rng.uniform(size=n_samples)

    if theta < 1e-10:
        # Independence
        return np.column_stack([u1, t])

    u2 = (u1 ** (-theta) * (t ** (-theta / (1.0 + theta)) - 1.0) + 1.0) ** (
        -1.0 / theta
    )
    return np.column_stack([u1, u2])


def _sample_gumbel(
    params: dict, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample bivariate Gumbel copula via Marshall-Olkin algorithm.

    Uses a stable distribution as the frailty variable.

    Algorithm (Marshall & Olkin, 1988):
        1. Sample V ~ Stable(1/theta, 1, cos(pi/(2*theta))^theta, 0)
        2. E1, E2 ~ Exp(1) independent
        3. u_j = exp(-(E_j / V)^{1/theta})
    """
    theta = params["theta"]

    if theta <= 1.0 + 1e-10:
        # Independence
        return np.column_stack(
            [rng.uniform(size=n_samples), rng.uniform(size=n_samples)]
        )

    alpha_stable = 1.0 / theta

    # Sample from stable distribution using Chambers-Mallows-Stuck method
    v_samples = _sample_stable(alpha_stable, n_samples, rng)

    e1 = rng.exponential(size=n_samples)
    e2 = rng.exponential(size=n_samples)

    u1 = np.exp(-((e1 / v_samples) ** alpha_stable))
    u2 = np.exp(-((e2 / v_samples) ** alpha_stable))

    return np.column_stack([u1, u2])


def _sample_stable(
    alpha: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample from a positive stable distribution S(alpha, 1, 0, 0).

    Chambers-Mallows-Stuck algorithm for totally skewed stable (beta=1).
    """
    if abs(alpha - 1.0) < 1e-10:
        return np.ones(n)

    phi = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    w = rng.exponential(size=n)

    # CMS formula for beta = 1
    term1 = np.sin(alpha * (phi + np.pi / 2)) / (np.cos(phi) ** (1.0 / alpha))
    term2 = (np.cos(phi - alpha * (phi + np.pi / 2)) / w) ** (
        (1.0 - alpha) / alpha
    )
    return term1 * term2


def _sample_frank(
    params: dict, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample bivariate Frank copula via conditional inversion.

    C_2|1(u2|u1) = t  =>  solve for u2.
    Conditional CDF: C_2|1(u2|u1) = exp(-theta*u1) / D

    where D = (exp(-theta*u1) - 1) / (exp(-theta*u2) - 1) + ...

    Closed form inversion:
        u2 = -1/theta * ln(1 + t*(exp(-theta) - 1) /
             (t + (1-t)*exp(-theta*u1)))
    """
    theta = params["theta"]
    u1 = rng.uniform(size=n_samples)
    t = rng.uniform(size=n_samples)

    if abs(theta) < 1e-10:
        return np.column_stack([u1, t])

    numerator = t * (np.exp(-theta) - 1.0)
    denominator = t + (1.0 - t) * np.exp(-theta * u1)
    u2 = -1.0 / theta * np.log(1.0 + numerator / denominator)
    u2 = np.clip(u2, 1e-10, 1 - 1e-10)

    return np.column_stack([u1, u2])


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------


def tail_dependence(params: dict) -> dict:
    """Compute analytic tail dependence coefficients.

    Parameters
    ----------
    params : dict
        Fitted copula parameters.

    Returns
    -------
    dict
        ``{"lambda_L": float, "lambda_U": float}`` -- lower and upper
        tail dependence coefficients.

    Notes
    -----
    - Gaussian: lambda_L = lambda_U = 0 (asymptotic independence).
    - Student-t: lambda_L = lambda_U = 2 * t_{df+1}(-sqrt((df+1)(1-rho)/(1+rho))).
    - Clayton:   lambda_L = 2^{-1/theta},  lambda_U = 0.
    - Gumbel:    lambda_L = 0,  lambda_U = 2 - 2^{1/theta}.
    - Frank:     lambda_L = lambda_U = 0.
    """
    family = params["family"]

    if family == "gaussian":
        return {"lambda_L": 0.0, "lambda_U": 0.0}

    elif family == "t":
        corr = params["corr_matrix"]
        df = params["df"]
        # Use the (0,1) entry as the representative correlation
        if corr.ndim == 2:
            rho = corr[0, 1]
        else:
            rho = float(corr)
        arg = -np.sqrt((df + 1) * (1.0 - rho) / (1.0 + rho))
        lambda_tail = 2.0 * stats.t.cdf(arg, df=df + 1)
        return {"lambda_L": float(lambda_tail), "lambda_U": float(lambda_tail)}

    elif family == "clayton":
        theta = params["theta"]
        if theta <= 0:
            return {"lambda_L": 0.0, "lambda_U": 0.0}
        lambda_L = 2.0 ** (-1.0 / theta)
        return {"lambda_L": float(lambda_L), "lambda_U": 0.0}

    elif family == "gumbel":
        theta = params["theta"]
        if theta <= 1.0:
            return {"lambda_L": 0.0, "lambda_U": 0.0}
        lambda_U = 2.0 - 2.0 ** (1.0 / theta)
        return {"lambda_L": 0.0, "lambda_U": float(lambda_U)}

    elif family == "frank":
        return {"lambda_L": 0.0, "lambda_U": 0.0}

    else:
        raise ValueError(f"Unknown copula family '{family}'.")
