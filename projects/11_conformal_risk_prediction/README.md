# 11 -- Conformal Risk Prediction

Distribution-free prediction intervals for financial risk measures with finite-sample coverage guarantees.

## Objectives

- Apply conformal prediction to construct coverage-guaranteed VaR and ES intervals.
- Implement the conformal risk control framework for user-specified risk functionals.
- Compare conformal intervals against parametric (normal, Student-t) and bootstrap approaches.
- Evaluate adaptive conformal methods under non-stationarity and distributional shift.

## Key Techniques

- Split conformal prediction with exchangeability assumptions
- Conformalized quantile regression for VaR intervals
- Conformal risk control (CRC) for monotone risk functionals
- Adaptive Conformal Inference (ACI) for time-series with distribution shift
- Weighted conformal prediction with importance-based covariate shift correction
- Coverage diagnostics: marginal, conditional, and stratified coverage rates
- Comparison with delta method, parametric bootstrap, and block bootstrap intervals

## Data Sources

- **yfinance** -- daily equity and index return series

## Dependencies

```
pip install "risk-analyst[ml]"
```

## References

1. Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal risk control. *Journal of Machine Learning Research*, 25(332), 1--47.
2. Luo, R. & Colombo, N. (2025). Conformal risk training. *NeurIPS 2025*.
3. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
