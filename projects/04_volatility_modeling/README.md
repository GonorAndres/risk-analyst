# 04 -- Volatility Modeling

GARCH-family and regime-switching models for conditional volatility estimation and forecasting.

## Objectives

- Estimate symmetric and asymmetric GARCH models (GARCH, GJR-GARCH, EGARCH, APARCH).
- Implement the Heterogeneous Autoregressive Realized Volatility (HAR-RV) model.
- Capture structural breaks with Hamilton Markov regime-switching models.
- Produce conditional VaR forecasts and compare model accuracy via loss functions.

## Key Techniques

- GARCH(1,1) with normal, Student-t, and skewed-t innovations
- GJR-GARCH and EGARCH for leverage effects
- APARCH (asymmetric power ARCH)
- HAR-RV model for realized volatility forecasting
- Markov regime-switching (2-state, 3-state) via maximum likelihood
- Volatility term structure construction
- Diebold-Mariano and Model Confidence Set for forecast comparison

## Data Sources

- **yfinance** -- S&P 500 returns, VIX index
- **CBOE** -- realized volatility and options-implied data

## Dependencies

```
pip install "risk-analyst[risk]"
```

## References

1. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987--1007.
2. Bollerslev, T. (1986). Generalized autoregressive conditional heteroscedasticity. *Journal of Econometrics*, 31(3), 307--327.
3. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357--384.
