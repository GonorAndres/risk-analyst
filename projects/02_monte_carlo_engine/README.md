# 02 -- Monte Carlo Simulation Engine

Reusable Monte Carlo engine for pricing derivatives and simulating correlated asset paths.

## Objectives

- Build a modular simulation framework supporting GBM and extensions (jump-diffusion, local vol).
- Generate correlated multi-asset paths via Cholesky decomposition.
- Implement variance reduction techniques to improve convergence.
- Price European, Asian, and barrier options with configurable payoff functions.

## Key Techniques

- Geometric Brownian Motion (GBM) discretization (Euler-Maruyama)
- Cholesky factorization for correlated Brownian increments
- Antithetic variates, control variates, and importance sampling
- Stratified sampling and quasi-Monte Carlo (Sobol sequences)
- Path-dependent option pricing (arithmetic Asian, knock-in/knock-out barriers)
- Optional GPU acceleration via CuPy / JAX for large-scale simulations

## Data Sources

- **yfinance** -- calibration of drift, volatility, and correlation from historical prices
- **QuantLib** -- benchmark analytical prices for validation

## Dependencies

```
pip install "risk-analyst[risk]"
```

## References

1. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
2. Broadie, M. & Glasserman, P. (1996). Estimating security price derivatives using simulation. *Management Science*, 42(2), 269--285.
3. Jackel, P. (2002). *Monte Carlo Methods in Finance*. Wiley.
