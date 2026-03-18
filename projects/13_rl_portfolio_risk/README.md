# 13 -- RL for Portfolio Risk Management

Reinforcement learning agents for dynamic portfolio allocation with explicit tail-risk constraints.

## Objectives

- Formulate portfolio risk management as a CVaR-constrained Markov Decision Process.
- Train and compare policy-gradient (PPO) and actor-critic (SAC, TD3) agents with risk-shaped rewards.
- Implement dynamic risk budgeting and tail risk parity allocation strategies.
- Benchmark RL policies against mean-variance, risk parity, and static CVaR-optimized portfolios.

## Key Techniques

- CVaR-constrained MDP formulation with Lagrangian relaxation
- Proximal Policy Optimization (PPO) with clipped objective
- Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3) for continuous action spaces
- Risk-sensitive reward shaping: Sharpe penalty, drawdown penalty, CVaR penalty
- Dynamic risk allocation via learned policy conditioned on regime indicators
- Tail risk parity: equalizing marginal CVaR contributions across assets
- Rolling-window backtesting with transaction cost modeling

## Data Sources

- **yfinance** -- multi-asset universe (equities, fixed income, commodities)

## Dependencies

```
pip install "risk-analyst[ml]"
```

## References

1. Wang, Y. et al. (2025). ICVaR-DRL: Iterated CVaR reinforcement learning for portfolio optimization. *Quantitative Finance*.
2. Tamar, A., Glassner, Y., & Mannor, S. (2015). Optimizing the CVaR via sampling. *AAAI Conference on Artificial Intelligence*.
3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
