"""Monte Carlo simulation engines and variance reduction techniques."""

from risk_analyst.simulation.gbm import simulate_gbm, simulate_gbm_correlated
from risk_analyst.simulation.option_pricing import (
    price_asian_option,
    price_barrier_option,
    price_european_option,
)
from risk_analyst.simulation.risk import mc_portfolio_es, mc_portfolio_var
from risk_analyst.simulation.variance_reduction import (
    antithetic_variates,
    control_variate,
    importance_sampling_var,
)

__all__ = [
    "simulate_gbm",
    "simulate_gbm_correlated",
    "antithetic_variates",
    "control_variate",
    "importance_sampling_var",
    "price_european_option",
    "price_asian_option",
    "price_barrier_option",
    "mc_portfolio_var",
    "mc_portfolio_es",
]
