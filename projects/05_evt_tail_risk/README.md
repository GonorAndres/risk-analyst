# 05 -- Extreme Value Theory for Tail Risk

EVT-based tail risk quantification using block maxima and peaks-over-threshold approaches.

## Objectives

- Fit Generalized Extreme Value (GEV) distributions via block maxima.
- Estimate Generalized Pareto Distribution (GPD) parameters using peaks-over-threshold.
- Develop threshold selection diagnostics (mean residual life plot, parameter stability plot).
- Compute EVT-based VaR and Expected Shortfall at 99.5% and 99.9% confidence levels.

## Key Techniques

- Block maxima method and GEV (Frechet, Gumbel, Weibull) fitting via MLE and PWM
- Peaks-over-threshold (POT) with GPD tail modeling
- Mean residual life plot and Hill estimator for threshold selection
- Non-stationary EVT with time-varying parameters
- Return level estimation and return period analysis
- Comparison of EVT vs normal and Student-t tail assumptions
- QQ-plots and Anderson-Darling goodness-of-fit tests

## Data Sources

- **yfinance** -- daily equity return series (long histories for tail estimation)

## Dependencies

```
pip install "risk-analyst[risk]"
```

## References

1. de Haan, L. & Ferreira, A. (2006). *Extreme Value Theory: An Introduction*. Springer.
2. McNeil, A. J. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. *Journal of Empirical Finance*, 7(3--4), 271--300.
3. Embrechts, P., Kluppelberg, C., & Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
