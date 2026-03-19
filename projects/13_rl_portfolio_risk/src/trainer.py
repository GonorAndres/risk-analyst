"""Training loop for RL portfolio agent using Natural Evolution Strategies.

Since the policy network is pure numpy (no autograd), we use NES with
mirror sampling for derivative-free optimisation. This follows the same
pattern as P08 deep hedging but adapted for the portfolio MDP setting.

Reference:
- Wierstra et al. (2014), "Natural Evolution Strategies", JMLR.
- Hansen (2016), "The CMA Evolution Strategy: A Tutorial".
- Wang et al. (2025), ICVaR-DRL for portfolio optimization.
"""

from __future__ import annotations

import numpy as np
from agent import PolicyNetwork
from environment import PortfolioEnv
from numpy.typing import NDArray


def compute_cvar(returns: NDArray[np.float64], alpha: float = 0.95) -> float:
    """Compute Conditional Value-at-Risk (Expected Shortfall) from returns.

    CVaR_alpha = E[-R | -R >= VaR_alpha]

    Parameters
    ----------
    returns : ndarray
        Array of portfolio returns (positive = gain).
    alpha : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    cvar : float
        CVaR value (positive means tail loss).
    """
    losses = -np.asarray(returns, dtype=np.float64)
    if len(losses) == 0:
        return 0.0
    var_level = float(np.percentile(losses, alpha * 100))
    tail = losses[losses >= var_level]
    if len(tail) == 0:
        return var_level
    return float(np.mean(tail))


def run_episode(
    agent: PolicyNetwork,
    env: PortfolioEnv,
    returns_data: NDArray[np.float64],
) -> dict:
    """Run a single episode of the portfolio MDP.

    Parameters
    ----------
    agent : PolicyNetwork
        Trained or untrained policy network.
    env : PortfolioEnv
        Portfolio environment.
    returns_data : ndarray of shape (T, n_assets)
        Return matrix for the episode.

    Returns
    -------
    results : dict
        Keys: total_return, sharpe, cvar, max_drawdown,
              weights_history, portfolio_returns, cumulative_returns.
    """
    state = env.reset(returns_data)
    done = False
    weights_history = []
    portfolio_returns = []
    total_reward = 0.0

    while not done:
        weights = agent.forward(state)
        next_state, reward, done, info = env.step(weights)
        weights_history.append(weights.copy())
        portfolio_returns.append(info["portfolio_return"])
        total_reward += reward
        state = next_state

    portfolio_returns_arr = np.array(portfolio_returns)
    weights_arr = np.array(weights_history)

    # Cumulative returns
    cumulative = np.cumprod(1.0 + portfolio_returns_arr)

    # Total return
    total_return = float(cumulative[-1] - 1.0) if len(cumulative) > 0 else 0.0

    # Sharpe ratio (annualised, assuming daily)
    if len(portfolio_returns_arr) > 1 and np.std(portfolio_returns_arr) > 1e-12:
        sharpe = float(
            np.mean(portfolio_returns_arr) / np.std(portfolio_returns_arr) * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    # CVaR at 95%
    cvar = compute_cvar(portfolio_returns_arr, alpha=0.95)

    # Maximum drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown_arr = (peak - cumulative) / np.maximum(peak, 1e-12)
    max_drawdown = float(np.max(drawdown_arr)) if len(drawdown_arr) > 0 else 0.0

    return {
        "total_return": total_return,
        "total_reward": total_reward,
        "sharpe": sharpe,
        "cvar": cvar,
        "max_drawdown": max_drawdown,
        "weights_history": weights_arr,
        "portfolio_returns": portfolio_returns_arr,
        "cumulative_returns": cumulative,
        "drawdown": drawdown_arr,
    }


def train_rl_agent(
    agent: PolicyNetwork,
    env: PortfolioEnv,
    returns_data: NDArray[np.float64],
    n_epochs: int = 100,
    population_size: int = 40,
    sigma: float = 0.02,
    lr: float = 0.01,
    cvar_lambda: float = 1.0,
    seed: int = 42,
) -> dict:
    """Train the RL agent using Natural Evolution Strategies with mirror sampling.

    At each epoch:
    1. Sample perturbation vectors.
    2. For each perturbation (and its mirror), run a full episode.
    3. Compute episode fitness = total reward.
    4. NES gradient estimate with mirror sampling.
    5. Update parameters toward higher-reward candidates.

    Parameters
    ----------
    agent : PolicyNetwork
        Policy network to optimise.
    env : PortfolioEnv
        Portfolio environment.
    returns_data : ndarray of shape (T, n_assets)
        Return matrix for training episodes.
    n_epochs : int
        Number of training epochs.
    population_size : int
        Number of perturbation directions per epoch.
    sigma : float
        Perturbation standard deviation.
    lr : float
        Learning rate for parameter updates.
    cvar_lambda : float
        Weight on CVaR penalty in the reward (stored in env).
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Keys: loss_history (negative rewards), final_reward, best_params.
    """
    rng = np.random.default_rng(seed)

    theta = agent.get_flat_params().copy()
    n_params = len(theta)

    loss_history: list[float] = []
    current_sigma = sigma

    for epoch in range(n_epochs):
        # Sample perturbation directions
        epsilon = rng.standard_normal((population_size, n_params))

        # Evaluate fitness for each perturbation (mirror sampling)
        fitness_plus = np.zeros(population_size)
        fitness_minus = np.zeros(population_size)

        for i in range(population_size):
            # Positive perturbation
            theta_plus = theta + current_sigma * epsilon[i]
            agent.set_flat_params(theta_plus)
            result_plus = run_episode(agent, env, returns_data)
            fitness_plus[i] = result_plus["total_reward"]

            # Negative perturbation (mirror)
            theta_minus = theta - current_sigma * epsilon[i]
            agent.set_flat_params(theta_minus)
            result_minus = run_episode(agent, env, returns_data)
            fitness_minus[i] = result_minus["total_reward"]

        # NES gradient estimate with mirror sampling:
        # grad ~= (1 / (2 * pop_size * sigma)) * sum((f+ - f-) * epsilon)
        # We want to MAXIMISE fitness, so we ascend
        advantages = fitness_plus - fitness_minus
        grad = np.mean(
            advantages[:, np.newaxis] * epsilon, axis=0
        ) / (2.0 * current_sigma)

        # Update parameters (gradient ASCENT on reward)
        theta = theta + lr * grad

        # Evaluate current fitness
        agent.set_flat_params(theta)
        current_result = run_episode(agent, env, returns_data)
        current_fitness = current_result["total_reward"]

        # Store negative reward as loss (for monotonicity check: loss should decrease)
        loss_history.append(-current_fitness)

        # Adaptive noise decay
        if epoch > 0 and epoch % 25 == 0:
            current_sigma = max(current_sigma * 0.9, 0.005)

    # Final params
    agent.set_flat_params(theta)

    return {
        "loss_history": loss_history,
        "final_reward": current_fitness,
        "best_params": theta.copy(),
    }
