"""Monte Carlo option pricing for European, Asian, and barrier options.

Each pricer returns (price, std_error, confidence_interval) where the
confidence interval is a 95 % symmetric interval around the price.

Validation helper ``bs_price`` provides the Black-Scholes closed-form
for European calls and puts.

Reference: Hull (2022), Chs. 15, 27; Glasserman (2003), Ch. 7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes closed-form (for validation)
# ---------------------------------------------------------------------------

def bs_price(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes closed-form price for a European option.

    Parameters
    ----------
    s0 : float
        Spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualised, continuously compounded).
    sigma : float
        Volatility (annualised).
    T : float
        Time to maturity in years.
    option_type : str
        ``"call"`` or ``"put"``.

    Returns
    -------
    price : float
    """
    if T <= 0:
        if option_type == "call":
            return max(s0 - K, 0.0)
        return max(K - s0, 0.0)

    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ---------------------------------------------------------------------------
# Monte Carlo pricers
# ---------------------------------------------------------------------------

def _mc_stats(
    payoffs: NDArray[np.float64],
    discount: float,
    z_crit: float = 1.96,
) -> tuple[float, float, tuple[float, float]]:
    """Compute price, standard error, and confidence interval from payoffs."""
    discounted = discount * payoffs
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    ci = (price - z_crit * std_error, price + z_crit * std_error)
    return price, std_error, ci


def price_european_option(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float, tuple[float, float]]:
    """Price a European call/put via Monte Carlo simulation.

    Simulates terminal price under risk-neutral measure:
        S_T = S_0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)

    Parameters
    ----------
    s0, K, r, sigma, T : float
        Standard Black-Scholes parameters.
    option_type : str
        ``"call"`` or ``"put"``.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int or None
        Random seed.

    Returns
    -------
    price : float
        Monte Carlo estimate of the option price.
    std_error : float
        Standard error of the estimate.
    confidence_interval : tuple[float, float]
        95 % confidence interval.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    # Risk-neutral terminal prices
    s_T = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    if option_type == "call":
        payoffs = np.maximum(s_T - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - s_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    discount = np.exp(-r * T)
    return _mc_stats(payoffs, discount)


def price_asian_option(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float, tuple[float, float]]:
    """Price an arithmetic Asian option via Monte Carlo.

    The payoff depends on the arithmetic average of prices along the path:
        payoff_call = max(A - K, 0),  where A = (1/n) * sum(S_ti)

    Parameters
    ----------
    s0, K, r, sigma, T : float
        Standard parameters.
    n_steps : int
        Number of monitoring dates.
    option_type : str
        ``"call"`` or ``"put"``.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int or None
        Random seed.

    Returns
    -------
    price, std_error, confidence_interval
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))

    # Risk-neutral path simulation
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    # Prices at each monitoring date (exclude t=0 from average per convention)
    paths = s0 * np.exp(log_paths)

    # Arithmetic average (over monitoring dates, not including t=0)
    avg_price = np.mean(paths, axis=1)

    if option_type == "call":
        payoffs = np.maximum(avg_price - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - avg_price, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    discount = np.exp(-r * T)
    return _mc_stats(payoffs, discount)


def price_barrier_option(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    barrier: float,
    barrier_type: str = "down-and-out",
    n_steps: int = 252,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float, tuple[float, float]]:
    """Price a barrier option (knock-out) via Monte Carlo.

    Supported barrier types:
      - ``"up-and-out"``:   option expires worthless if S ever exceeds barrier.
      - ``"down-and-out"``: option expires worthless if S ever falls below barrier.

    The underlying is a European call.

    Parameters
    ----------
    s0, K, r, sigma, T : float
        Standard parameters.
    barrier : float
        Barrier level.
    barrier_type : str
        ``"up-and-out"`` or ``"down-and-out"``.
    n_steps : int
        Number of monitoring dates for barrier checking.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int or None
        Random seed.

    Returns
    -------
    price, std_error, confidence_interval
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))

    # Build full price paths (including t=0)
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(log_increments, axis=1)
    paths = np.exp(log_paths)

    # Determine which paths survive (barrier not breached)
    if barrier_type == "up-and-out":
        survived = np.all(paths <= barrier, axis=1)
    elif barrier_type == "down-and-out":
        survived = np.all(paths >= barrier, axis=1)
    else:
        raise ValueError(
            f"barrier_type must be 'up-and-out' or 'down-and-out', got '{barrier_type}'"
        )

    # Terminal payoff (European call) for surviving paths only
    s_T = paths[:, -1]
    payoffs = np.where(survived, np.maximum(s_T - K, 0.0), 0.0)

    discount = np.exp(-r * T)
    return _mc_stats(payoffs, discount)
