"""VaR backtesting framework.

Implements the standard regulatory and statistical backtests:
    1. Kupiec (1995) proportion-of-failures test (unconditional coverage)
    2. Christoffersen (1998) conditional coverage test (independence + coverage)
    3. Basel traffic-light test (green / yellow / red zones)

References:
    - Kupiec (1995): "Techniques for Verifying the Accuracy of Risk
      Measurement Models", *Journal of Derivatives*.
    - Christoffersen (1998): "Evaluating Interval Forecasts",
      *International Economic Review*.
    - Basel Committee (1996): "Supervisory Framework for Backtesting".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Container for backtesting test outputs."""

    statistic: float
    p_value: float
    reject: bool


@dataclass
class TrafficLightResult:
    """Container for Basel traffic-light test output."""

    zone: Literal["green", "yellow", "red"]
    n_violations: int
    n_obs: int


@dataclass
class BacktestReport:
    """Aggregate backtesting report."""

    n_violations: int
    n_obs: int
    violation_rate: float
    expected_rate: float
    kupiec: TestResult
    christoffersen: TestResult
    traffic_light: TrafficLightResult


def kupiec_test(
    violations: int,
    n_obs: int,
    alpha: float,
    significance: float = 0.05,
) -> TestResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    Tests H_0: the true violation rate equals (1 - alpha).

    LR_uc = -2 * ln[ (1-p)^{n-x} p^x / (1-hat{p})^{n-x} hat{p}^x ]

    where p = 1 - alpha (expected rate), hat{p} = x / n (observed rate),
    x = number of violations, n = number of observations.

    Under H_0, LR_uc ~ chi-squared(1).

    Parameters
    ----------
    violations : int
        Number of VaR exceedances.
    n_obs : int
        Total number of backtesting observations.
    alpha : float
        VaR confidence level (e.g. 0.99).
    significance : float
        Test significance level for the reject decision.

    Returns
    -------
    TestResult
        LR statistic, p-value, and reject flag.
    """
    p = 1.0 - alpha  # expected violation rate
    p_hat = violations / n_obs if n_obs > 0 else 0.0

    # Avoid log(0) edge cases
    if violations == 0:
        # p_hat = 0: L1 = ln(1^n) = 0, L0 = n*ln(1-p)
        # LR = -2*(L0 - L1) = -2*n*ln(1-p)
        lr = -2.0 * n_obs * np.log(1.0 - p)
    elif violations == n_obs:
        lr = -2.0 * n_obs * np.log(p)
    else:
        log_l0 = (n_obs - violations) * np.log(1.0 - p) + violations * np.log(p)
        log_l1 = (n_obs - violations) * np.log(1.0 - p_hat) + violations * np.log(p_hat)
        lr = -2.0 * (log_l0 - log_l1)

    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))
    return TestResult(statistic=float(lr), p_value=p_value, reject=p_value < significance)


def christoffersen_test(
    violations_series: np.ndarray,
    significance: float = 0.05,
) -> TestResult:
    """Christoffersen (1998) conditional coverage test.

    Combines the unconditional coverage test with an independence test
    for the violation indicator sequence.

    The independence LR is based on a first-order Markov chain for the
    binary violation process I_t in {0, 1}:

    LR_ind = -2 * ln[ L(pi_hat) / L(pi_01, pi_11) ]

    where pi_ij = P(I_t=j | I_{t-1}=i) estimated from transitions.

    Under H_0 (independent violations), LR_ind ~ chi-squared(1).

    Parameters
    ----------
    violations_series : np.ndarray
        Binary array of length T: 1 = violation, 0 = no violation.
    significance : float
        Test significance level.

    Returns
    -------
    TestResult
        LR statistic (independence component), p-value, and reject flag.
    """
    v = np.asarray(violations_series, dtype=int)
    T = len(v)

    if T < 2:
        return TestResult(statistic=0.0, p_value=1.0, reject=False)

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if v[t - 1] == 0 and v[t] == 0:
            n00 += 1
        elif v[t - 1] == 0 and v[t] == 1:
            n01 += 1
        elif v[t - 1] == 1 and v[t] == 0:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    n0 = n00 + n01  # transitions from state 0
    n1 = n10 + n11  # transitions from state 1

    # Handle degenerate cases
    if n0 == 0 or n1 == 0:
        return TestResult(statistic=0.0, p_value=1.0, reject=False)

    pi_01 = n01 / n0 if n0 > 0 else 0.0
    pi_11 = n11 / n1 if n1 > 0 else 0.0
    pi_hat = (n01 + n11) / (T - 1)  # unconditional estimate

    # Log-likelihood under independence (H_0)
    eps = 1e-15
    log_l0 = (n00 + n10) * np.log(max(1.0 - pi_hat, eps)) + (n01 + n11) * np.log(
        max(pi_hat, eps)
    )

    # Log-likelihood under Markov dependence (H_1)
    log_l1 = 0.0
    if n00 > 0:
        log_l1 += n00 * np.log(max(1.0 - pi_01, eps))
    if n01 > 0:
        log_l1 += n01 * np.log(max(pi_01, eps))
    if n10 > 0:
        log_l1 += n10 * np.log(max(1.0 - pi_11, eps))
    if n11 > 0:
        log_l1 += n11 * np.log(max(pi_11, eps))

    lr_ind = -2.0 * (log_l0 - log_l1)
    lr_ind = max(lr_ind, 0.0)  # numerical guard

    p_value = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    return TestResult(statistic=float(lr_ind), p_value=p_value, reject=p_value < significance)


def traffic_light_test(
    n_violations: int,
    n_obs: int,
) -> TrafficLightResult:
    """Basel traffic-light backtesting classification.

    For a 250-day test window at 99% confidence (expected = 2.5 violations):
        - Green:  0--4 violations
        - Yellow: 5--9 violations
        - Red:    10+ violations

    The thresholds are scaled proportionally for other window sizes,
    using the binomial distribution.

    Parameters
    ----------
    n_violations : int
        Observed number of VaR exceedances.
    n_obs : int
        Number of backtesting observations.

    Returns
    -------
    TrafficLightResult
        Zone classification and counts.
    """
    # Basel thresholds are defined for 250-day window at 99% VaR.
    # Scale: green up to ~1.6% violation rate, yellow up to ~3.6%, red above.
    # Standard Basel thresholds (250 obs): green <= 4, yellow 5-9, red >= 10.
    scale = n_obs / 250.0
    green_max = int(4 * scale)
    yellow_max = int(9 * scale)

    if n_violations <= green_max:
        zone: Literal["green", "yellow", "red"] = "green"
    elif n_violations <= yellow_max:
        zone = "yellow"
    else:
        zone = "red"

    return TrafficLightResult(zone=zone, n_violations=n_violations, n_obs=n_obs)


def backtest_var(
    losses: np.ndarray,
    var_series: np.ndarray,
    alpha: float,
    kupiec_significance: float = 0.05,
    christoffersen_significance: float = 0.05,
) -> BacktestReport:
    """Run the full backtesting suite on a VaR time series.

    Compares realized losses against the VaR forecast for each period
    and runs Kupiec, Christoffersen, and traffic-light tests.

    Parameters
    ----------
    losses : np.ndarray
        Realized portfolio losses (T,).
    var_series : np.ndarray
        VaR forecasts (T,), same length as *losses*.
    alpha : float
        VaR confidence level.
    kupiec_significance : float
        Significance level for Kupiec test.
    christoffersen_significance : float
        Significance level for Christoffersen test.

    Returns
    -------
    BacktestReport
        Consolidated backtesting report.
    """
    losses = np.asarray(losses)
    var_series = np.asarray(var_series)

    violations_indicator = (losses >= var_series).astype(int)
    n_violations = int(violations_indicator.sum())
    n_obs = len(losses)
    violation_rate = n_violations / n_obs if n_obs > 0 else 0.0
    expected_rate = 1.0 - alpha

    kupiec = kupiec_test(n_violations, n_obs, alpha, kupiec_significance)
    christoffersen = christoffersen_test(violations_indicator, christoffersen_significance)
    tl = traffic_light_test(n_violations, n_obs)

    return BacktestReport(
        n_violations=n_violations,
        n_obs=n_obs,
        violation_rate=violation_rate,
        expected_rate=expected_rate,
        kupiec=kupiec,
        christoffersen=christoffersen,
        traffic_light=tl,
    )
