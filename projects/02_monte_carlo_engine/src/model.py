"""MonteCarloEngine -- orchestrator for Project 02.

Wraps the shared ``risk_analyst.simulation`` library to provide a
high-level interface for portfolio simulation, risk computation,
option pricing, and variance-reduction comparison.

All parameters are read from a YAML config (see configs/default.yaml).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from risk_analyst.simulation.gbm import simulate_gbm_correlated
from risk_analyst.simulation.option_pricing import (
    bs_price,
    price_asian_option,
    price_barrier_option,
    price_european_option,
)
from risk_analyst.simulation.risk import mc_portfolio_es, mc_portfolio_var


class MonteCarloEngine:
    """High-level Monte Carlo engine driven by a YAML config.

    Parameters
    ----------
    config : dict or str or Path
        If a dict, used directly.  If a str/Path, loaded from YAML file.
    """

    def __init__(self, config: dict[str, Any] | str | Path) -> None:
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config: dict[str, Any] = yaml.safe_load(f)
        else:
            self.config = config

        # Unpack simulation settings with defaults
        sim = self.config.get("simulation", {})
        self.n_paths: int = sim.get("n_paths", 10_000)
        self.n_steps: int = sim.get("n_steps", 252)
        self.seed: int | None = sim.get("seed", 42)

        # Variance reduction flags
        vr = self.config.get("variance_reduction", {})
        self.use_antithetic: bool = vr.get("use_antithetic", False)
        self.use_control_variate: bool = vr.get("use_control_variate", False)

        # Option pricing defaults
        op = self.config.get("option_pricing", {})
        self.r: float = op.get("risk_free_rate", 0.05)

        # Portfolio settings
        pf = self.config.get("portfolio", {})
        self.tickers: list[str] = pf.get("tickers", [])
        self.weights: list[float] = pf.get("weights", [])
        self.confidence_levels: list[float] = pf.get("confidence_levels", [0.95, 0.99])

        # Cached simulation results
        self._simulated_paths: NDArray[np.float64] | None = None
        self._portfolio_returns: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Portfolio simulation
    # ------------------------------------------------------------------

    def simulate_portfolio(
        self,
        prices: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Simulate future portfolio values from historical prices.

        Estimates per-asset drift and volatility from ``prices``, then
        runs correlated GBM forward for ``n_steps`` steps.

        Parameters
        ----------
        prices : array of shape (n_obs, n_assets)
            Historical price matrix (rows = dates, columns = assets).
        weights : array of shape (n_assets,) or None
            Portfolio weights.  If None, uses config weights.

        Returns
        -------
        portfolio_paths : array of shape (n_paths, n_steps + 1)
            Simulated portfolio value paths (starting at 1.0).
        """
        prices = np.asarray(prices, dtype=np.float64)
        if weights is None:
            weights = np.asarray(self.weights, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)

        # Estimate parameters from historical log-returns
        log_returns = np.diff(np.log(prices), axis=0)  # (n_obs-1, n_assets)
        n_assets = prices.shape[1]

        # Annualise (assuming 252 trading days)
        mu_vec = np.mean(log_returns, axis=0) * 252
        sigma_vec = np.std(log_returns, axis=0, ddof=1) * np.sqrt(252)
        corr_matrix = np.corrcoef(log_returns, rowvar=False)

        # Handle single-asset edge case
        if n_assets == 1:
            corr_matrix = np.array([[1.0]])

        # Normalise prices to start at 1
        s0_vec = np.ones(n_assets)

        # Simulate 1-year forward
        T = self.n_steps / 252.0
        asset_paths = simulate_gbm_correlated(
            s0_vec=s0_vec,
            mu_vec=mu_vec,
            sigma_vec=sigma_vec,
            corr_matrix=corr_matrix,
            T=T,
            n_steps=self.n_steps,
            n_paths=self.n_paths,
            seed=self.seed,
        )  # (n_paths, n_steps+1, n_assets)

        # Portfolio value = weighted sum of asset paths
        portfolio_paths = np.tensordot(asset_paths, weights, axes=([-1], [0]))
        # shape: (n_paths, n_steps+1)

        self._simulated_paths = asset_paths
        self._portfolio_returns = np.diff(np.log(portfolio_paths), axis=1)

        return portfolio_paths

    # ------------------------------------------------------------------
    # Risk computation
    # ------------------------------------------------------------------

    def compute_risk(
        self,
        alpha: float | list[float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute VaR and ES from simulated portfolio paths.

        Must call ``simulate_portfolio`` first, or pass historical
        returns directly via the underlying functions.

        Parameters
        ----------
        alpha : float, list of floats, or None
            Confidence level(s).  If None, uses config levels.

        Returns
        -------
        results : dict
            ``{alpha_str: {"VaR": ..., "ES": ...}}``
        """
        if self._portfolio_returns is None:
            raise RuntimeError("Call simulate_portfolio() before compute_risk().")

        if alpha is None:
            alphas = self.confidence_levels
        elif isinstance(alpha, (int, float)):
            alphas = [float(alpha)]
        else:
            alphas = [float(a) for a in alpha]

        # Reshape portfolio returns to (n_sims, 1) for the risk functions
        pr = self._portfolio_returns  # (n_paths, n_steps)
        # Use all per-step returns as the sample
        flat_returns = pr.reshape(-1, 1)  # (n_paths * n_steps, 1)
        w = np.array([1.0])

        results: dict[str, dict[str, float]] = {}
        for a in alphas:
            var = mc_portfolio_var(flat_returns, w, alpha=a, n_sims=len(flat_returns), seed=self.seed)
            es = mc_portfolio_es(flat_returns, w, alpha=a, n_sims=len(flat_returns), seed=self.seed)
            results[f"{a:.2f}"] = {"VaR": var, "ES": es}

        return results

    # ------------------------------------------------------------------
    # Option pricing
    # ------------------------------------------------------------------

    def price_option(
        self,
        option_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Price a derivative instrument.

        Parameters
        ----------
        option_params : dict
            Must contain:
              - ``"style"``: ``"european"``, ``"asian"``, or ``"barrier"``
              - ``"s0"``, ``"K"``, ``"sigma"``, ``"T"``
              - ``"option_type"``: ``"call"`` or ``"put"``
            For barrier options additionally:
              - ``"barrier"``, ``"barrier_type"``

        Returns
        -------
        result : dict
            Contains ``"price"``, ``"std_error"``, ``"ci"``,
            and for European options also ``"bs_price"`` for comparison.
        """
        style = option_params.get("style", "european")
        s0 = option_params["s0"]
        K = option_params["K"]
        sigma = option_params["sigma"]
        T = option_params["T"]
        r = option_params.get("r", self.r)
        option_type = option_params.get("option_type", "call")
        n_paths = option_params.get("n_paths", self.n_paths)
        seed = option_params.get("seed", self.seed)

        result: dict[str, Any] = {"style": style, "option_type": option_type}

        if style == "european":
            price, se, ci = price_european_option(
                s0=s0, K=K, r=r, sigma=sigma, T=T,
                option_type=option_type, n_paths=n_paths, seed=seed,
            )
            result["bs_price"] = bs_price(s0, K, r, sigma, T, option_type)

        elif style == "asian":
            n_steps = option_params.get("n_steps", self.n_steps)
            price, se, ci = price_asian_option(
                s0=s0, K=K, r=r, sigma=sigma, T=T,
                n_steps=n_steps, option_type=option_type,
                n_paths=n_paths, seed=seed,
            )

        elif style == "barrier":
            barrier = option_params["barrier"]
            barrier_type = option_params.get("barrier_type", "down-and-out")
            n_steps = option_params.get("n_steps", self.n_steps)
            price, se, ci = price_barrier_option(
                s0=s0, K=K, r=r, sigma=sigma, T=T,
                barrier=barrier, barrier_type=barrier_type,
                n_steps=n_steps, n_paths=n_paths, seed=seed,
            )
        else:
            raise ValueError(f"Unknown option style: {style}")

        result["price"] = price
        result["std_error"] = se
        result["ci"] = ci

        return result

    # ------------------------------------------------------------------
    # Variance reduction comparison
    # ------------------------------------------------------------------

    def compare_variance_reduction(
        self,
        s0: float = 100.0,
        K: float = 100.0,
        sigma: float = 0.2,
        T: float = 1.0,
        option_type: str = "call",
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Compare naive MC vs antithetic variates for European option pricing.

        Runs the same pricing problem twice -- once with plain MC and once
        with antithetic variates -- and reports the efficiency gain.

        Parameters
        ----------
        s0, K, sigma, T : float
            Option parameters.
        option_type : str
            ``"call"`` or ``"put"``.
        n_paths : int or None
            Overrides config n_paths if given.
        seed : int or None
            Random seed.

        Returns
        -------
        comparison : dict
            Contains naive and antithetic prices, standard errors,
            and the variance reduction ratio.
        """
        if n_paths is None:
            n_paths = self.n_paths
        if seed is None:
            seed = self.seed
        r = self.r

        # --- Naive MC ---
        naive_price, naive_se, naive_ci = price_european_option(
            s0=s0, K=K, r=r, sigma=sigma, T=T,
            option_type=option_type, n_paths=n_paths, seed=seed,
        )

        # --- Antithetic variates ---
        # Generate Z and -Z paths, price each, average
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_paths)

        discount = np.exp(-r * T)
        s_T_pos = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        s_T_neg = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-z))

        if option_type == "call":
            payoff_pos = np.maximum(s_T_pos - K, 0.0)
            payoff_neg = np.maximum(s_T_neg - K, 0.0)
        else:
            payoff_pos = np.maximum(K - s_T_pos, 0.0)
            payoff_neg = np.maximum(K - s_T_neg, 0.0)

        # Antithetic estimate per pair
        payoff_avg = (payoff_pos + payoff_neg) / 2.0
        anti_discounted = discount * payoff_avg
        anti_price = float(np.mean(anti_discounted))
        anti_se = float(np.std(anti_discounted, ddof=1) / np.sqrt(n_paths))
        anti_ci = (anti_price - 1.96 * anti_se, anti_price + 1.96 * anti_se)

        # Analytical benchmark
        bs = bs_price(s0, K, r, sigma, T, option_type)

        # Variance reduction ratio
        vr_ratio = (naive_se / anti_se) ** 2 if anti_se > 1e-15 else float("inf")

        return {
            "bs_price": bs,
            "naive": {"price": naive_price, "std_error": naive_se, "ci": naive_ci},
            "antithetic": {"price": anti_price, "std_error": anti_se, "ci": anti_ci},
            "variance_reduction_ratio": vr_ratio,
        }
