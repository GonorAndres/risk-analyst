# 08 -- Deep Hedging

Neural network-based hedging strategies that minimize risk-adjusted hedging error for derivatives.

## Objectives

- Train feed-forward and recurrent networks to learn hedging policies for European and exotic options.
- Optimize hedging under CVaR and other risk-measure objectives (beyond mean-variance).
- Incorporate realistic frictions: proportional and fixed transaction costs, discrete rebalancing.
- Benchmark learned strategies against Black-Scholes delta hedging and delta-gamma hedging.

## Key Techniques

- Deep hedging architecture: policy network mapping market state to hedge ratios
- CVaR (Conditional Value-at-Risk) as the training loss function
- Model-free pricing as the initial value of the optimal hedging portfolio
- Recurrent networks (LSTM, GRU) for path-dependent payoffs
- Transaction cost penalization in the objective
- GBM and Heston stochastic volatility simulators for training data
- Convergence diagnostics and out-of-sample P&L analysis

## Data Sources

- **Simulated paths** -- GBM, Heston, and SABR model-generated price trajectories

## Dependencies

```
pip install "risk-analyst[risk,ml]"
```

## References

1. Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance*, 19(8), 1271--1291.
2. Ruf, J. & Wang, W. (2020). Neural networks for option pricing and hedging: a literature review. *Journal of Computational Finance*, 24(1).
3. Murray, P., Wood, B., Buehler, H., Wiese, M., & Pham, M. (2025). Deep hedging with market impact. *Quantitative Finance*.
