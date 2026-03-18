# 06 -- Copula Dependency Modeling

Copula-based dependence structures for multi-asset risk aggregation and scenario generation.

## Objectives

- Filter marginals through GARCH models and transform to uniform via the probability integral transform.
- Estimate and compare elliptical (Gaussian, Student-t) and Archimedean (Clayton, Gumbel, Frank) copulas.
- Construct vine copulas (R-vine) for high-dimensional dependence.
- Generate joint scenarios and compute portfolio VaR under different copula assumptions.

## Key Techniques

- GARCH-filtered standardized residuals for marginal modeling
- Probability integral transform (PIT) and Rosenblatt transform
- Maximum likelihood estimation (IFM and full MLE) for copula parameters
- Tail dependence coefficients (upper and lower)
- R-vine, C-vine, and D-vine copula decompositions
- Goodness-of-fit via Cramer-von Mises and Rosenblatt-based tests
- Copula-based Monte Carlo simulation for joint loss scenarios

## Data Sources

- **yfinance** -- multi-asset universe (equities, bonds, commodities, FX)

## Dependencies

```
pip install "risk-analyst[risk]"
```

## References

1. Nelsen, R. B. (2006). *An Introduction to Copulas*. 2nd ed. Springer.
2. Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC.
3. Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics*, 44(2), 182--198.
