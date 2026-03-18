# 01 -- Portfolio Risk Dashboard

Interactive dashboard for computing, visualizing, and backtesting portfolio risk measures.

## Objectives

- Compute VaR (parametric, historical simulation, Monte Carlo) and CVaR at multiple confidence levels.
- Track maximum drawdown, Sharpe ratio, and Sortino ratio over configurable horizons.
- Backtest VaR models using Kupiec, Christoffersen, and Basel traffic-light tests.
- Display rolling volatility surfaces and correlation heatmaps via an interactive Streamlit UI.

## Key Techniques

- Parametric VaR (variance-covariance with normal and Student-t assumptions)
- Historical simulation VaR and filtered historical simulation
- Monte Carlo VaR with full revaluation
- Expected Shortfall (CVaR) as a coherent risk measure
- Kupiec proportion-of-failures test (unconditional coverage)
- Christoffersen interval-forecast test (conditional coverage and independence)
- Basel traffic-light backtesting framework (green / yellow / red zones)
- Exponentially weighted moving average (EWMA) for rolling volatility and correlation

## Data Sources

- **yfinance** -- daily adjusted close prices for equity portfolios

## Dependencies

```
pip install "risk-analyst[risk,viz]"
```

## References

1. Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk*. 3rd ed. McGraw-Hill.
2. Basel Committee on Banking Supervision (1996). *Supervisory Framework for the Use of "Backtesting" in Conjunction with the Internal Models Approach to Market Risk Capital Requirements*.
3. Acerbi, C. & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance*, 26(7), 1487--1503.
