# 07 -- Stress Testing Framework

Multi-scenario stress testing engine with macro-financial transmission and reverse stress testing.

## Objectives

- Implement Fed DFAST scenario architecture (baseline, adverse, severely adverse).
- Build macro-to-portfolio transmission models via multivariate regression.
- Estimate credit migration matrices under stress conditions.
- Replicate historical stress episodes (2008 GFC, COVID-19, SVB 2023) and perform reverse stress testing.

## Key Techniques

- Scenario design: deterministic (regulatory) and stochastic (Monte Carlo) stress paths
- Macro factor regression (GDP, unemployment, credit spreads) to portfolio P&L
- Credit transition matrix estimation and stressed migration via generator matrices
- Historical scenario replay with mark-to-market revaluation
- Reverse stress testing: optimization to find scenarios breaching loss thresholds
- Sensitivity and concentration risk analysis
- Multi-period stress propagation with feedback effects

## Data Sources

- **FRED** -- macroeconomic time series (GDP, unemployment, spreads)
- **Fed DFAST/CCAR** -- supervisory scenario variables
- **EBA** -- European stress test scenario data

## Dependencies

```
pip install "risk-analyst[risk,ml]"
```

## References

1. Basel Committee on Banking Supervision (2018). *Stress Testing Principles*.
2. Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73--84.
3. Breuer, T. & Csiszar, I. (2013). Systematic stress tests with entropic plausibility constraints. *Journal of Banking & Finance*, 37(5), 1552--1559.
