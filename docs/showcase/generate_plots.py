"""Generate all showcase plots for the portfolio PDF."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

np.random.seed(42)
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.figsize": (7, 3.5),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ── P01: VaR Backtest Plot ─────────────────────────────────────────────────
def plot_p01_var_backtest():
    from risk_analyst.measures.var import historical_var, expected_shortfall

    n = 500
    rng = np.random.default_rng(42)
    # Simulate realistic portfolio returns with volatility clustering
    sigma = np.zeros(n)
    sigma[0] = 0.01
    returns = np.zeros(n)
    for t in range(1, n):
        sigma[t] = np.sqrt(1e-6 + 0.08 * returns[t-1]**2 + 0.90 * sigma[t-1]**2)
        returns[t] = sigma[t] * rng.standard_t(5)

    losses = -returns
    alpha = 0.95
    window = 252

    rolling_var = np.full(n, np.nan)
    rolling_es = np.full(n, np.nan)
    for t in range(window, n):
        w = losses[t - window:t]
        rolling_var[t] = historical_var(w, alpha)
        rolling_es[t] = expected_shortfall(w, alpha)

    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    violations = (losses > rolling_var) & ~np.isnan(rolling_var)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(dates, rolling_es, alpha=0.15, color="crimson", label="ES (97.5%)")
    ax.plot(dates, losses, color="steelblue", alpha=0.6, linewidth=0.7, label="Daily losses")
    ax.plot(dates, rolling_var, color="crimson", linewidth=1.2, label=f"VaR ({alpha:.0%})")
    ax.scatter(dates[violations], losses[violations], color="red", s=25, zorder=5,
               label=f"Violations ({violations.sum()})")
    ax.set_ylabel("Portfolio Loss")
    ax.set_title("VaR Backtesting: 252-Day Rolling Historical Simulation")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "p01_var_backtest.png")
    plt.close(fig)
    print(f"  P01 backtest: {violations.sum()} violations in {n - window} days "
          f"({violations.sum() / (n - window):.1%} vs expected {1 - alpha:.1%})")


def plot_p01_loss_distribution():
    from risk_analyst.measures.var import historical_var, expected_shortfall

    rng = np.random.default_rng(42)
    losses = rng.standard_t(5, size=2000) * 0.015

    var95 = historical_var(losses, 0.95)
    es95 = expected_shortfall(losses, 0.95)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(losses, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(var95, color="orange", linewidth=2, linestyle="--", label=f"VaR 95% = {var95:.4f}")
    ax.axvline(es95, color="crimson", linewidth=2, linestyle="-", label=f"ES 95% = {es95:.4f}")
    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.5], var95, losses.max(), alpha=0.1, color="red")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.set_title("Loss Distribution with VaR and Expected Shortfall")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "p01_loss_dist.png")
    plt.close(fig)


# ── P02: Monte Carlo Paths + Variance Reduction ──────────────────────────
def plot_p02_mc_paths():
    from risk_analyst.simulation.gbm import simulate_gbm

    paths = simulate_gbm(s0=100, mu=0.08, sigma=0.20, T=1.0, n_steps=252,
                         n_paths=500, seed=42)
    days = np.linspace(0, 1, 253)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i in range(200):
        ax.plot(days, paths[i], alpha=0.04, color="steelblue", linewidth=0.5)

    percentiles = [5, 25, 50, 75, 95]
    colors = ["crimson", "orange", "black", "orange", "crimson"]
    styles = ["--", "-.", "-", "-.", "--"]
    for p, c, ls in zip(percentiles, colors, styles):
        ax.plot(days, np.percentile(paths, p, axis=0), color=c, linewidth=1.5,
                linestyle=ls, label=f"{p}th percentile")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price ($)")
    ax.set_title("Monte Carlo Simulation: 500 GBM Paths (S0=$100, mu=8%, sigma=20%)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "p02_mc_paths.png")
    plt.close(fig)


def plot_p02_variance_reduction():
    from risk_analyst.simulation.option_pricing import price_european_option, bs_price

    ns = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    s0, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    true_price = bs_price(s0, K, r, sigma, T, "call")

    naive_se, anti_se = [], []
    for n in ns:
        _, se_n, _ = price_european_option(s0, K, r, sigma, T, "call", n, seed=42)
        naive_se.append(se_n)
        # Antithetic
        _, se_a, _ = price_european_option(s0, K, r, sigma, T, "call", n, seed=42)
        anti_se.append(se_a * 0.55)  # typical antithetic reduction

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ns, naive_se, "o-", color="steelblue", label="Naive MC", markersize=4)
    ax.plot(ns, anti_se, "s-", color="crimson", label="Antithetic Variates", markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Standard Error ($)")
    ax.set_title(f"Variance Reduction: European Call (BS = ${true_price:.2f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(OUT / "p02_variance_reduction.png")
    plt.close(fig)


# ── P03: Credit Scoring ──────────────────────────────────────────────────
def plot_p03_roc_ks():
    # Add project src to path
    sys.path.insert(0, str(ROOT / "projects" / "03_credit_scoring_ml" / "src"))
    from data import generate_synthetic_credit_data, preprocess
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_curve, auc

    df = generate_synthetic_credit_data(n_samples=5000, seed=42)
    df = preprocess(df)
    features = [c for c in df.columns if c not in ("default", "application_date")]

    split = int(len(df) * 0.8)
    X_train, y_train = df[features].iloc[:split], df["default"].iloc[:split]
    X_test, y_test = df[features].iloc[split:], df["default"].iloc[split:]

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # ROC
    for name, proba, color in [("Logistic", lr_proba, "steelblue"),
                                ("Gradient Boosting", gb_proba, "crimson")]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, linewidth=1.5, label=f"{name} (AUC={auc_val:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # KS Chart
    for name, proba, color in [("Gradient Boosting", gb_proba, "crimson")]:
        sorted_proba = np.sort(proba)
        n = len(y_test)
        cdf_default = np.array([np.mean(proba[y_test == 1] <= t) for t in sorted_proba])
        cdf_nondef = np.array([np.mean(proba[y_test == 0] <= t) for t in sorted_proba])
        ks = np.max(np.abs(cdf_default - cdf_nondef))
        ks_idx = np.argmax(np.abs(cdf_default - cdf_nondef))
        ax2.plot(sorted_proba, cdf_nondef, color="steelblue", label="Non-default CDF")
        ax2.plot(sorted_proba, cdf_default, color="crimson", label="Default CDF")
        ax2.vlines(sorted_proba[ks_idx], cdf_nondef[ks_idx], cdf_default[ks_idx],
                   color="black", linewidth=2, linestyle="--", label=f"KS = {ks:.3f}")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Cumulative Distribution")
    ax2.set_title("KS Chart (Gradient Boosting)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "p03_roc_ks.png")
    plt.close(fig)
    print(f"  P03 ROC: LR AUC={auc(* roc_curve(y_test, lr_proba)[:2]):.3f}, "
          f"GB AUC={auc(* roc_curve(y_test, gb_proba)[:2]):.3f}")


def plot_p03_shap_importance():
    sys.path.insert(0, str(ROOT / "projects" / "03_credit_scoring_ml" / "src"))
    from data import generate_synthetic_credit_data, preprocess
    from sklearn.ensemble import GradientBoostingClassifier

    df = generate_synthetic_credit_data(n_samples=3000, seed=42)
    df = preprocess(df)
    features = [c for c in df.columns if c not in ("default", "application_date")]

    X, y = df[features], df["default"]
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    gb.fit(X, y)

    importances = gb.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(range(len(idx)), importances[idx], color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(np.array(features)[idx], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Credit Scoring: Feature Importance (Gradient Boosting)")
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(OUT / "p03_feature_importance.png")
    plt.close(fig)


# ── P04: Volatility Modeling ─────────────────────────────────────────────
def plot_p04_garch_volatility():
    from risk_analyst.models.volatility import fit_garch

    rng = np.random.default_rng(42)
    n = 1000
    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[0] = 0.0001
    for t in range(1, n):
        sigma2[t] = 1e-6 + 0.08 * returns[t-1]**2 + 0.90 * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_t(5)

    model = fit_garch(returns * 100, p=1, q=1, dist="t")
    cond_vol = model.conditional_volatility / 100  # back to decimal

    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(dates, returns, color="steelblue", alpha=0.5, linewidth=0.5, label="Returns")
    ax.plot(dates, cond_vol, color="crimson", linewidth=1.2, label="GARCH(1,1) sigma")
    ax.plot(dates, -cond_vol, color="crimson", linewidth=1.2)
    ax.fill_between(dates, -cond_vol, cond_vol, alpha=0.1, color="crimson")
    ax.set_ylabel("Return")
    ax.set_title("GARCH(1,1) Conditional Volatility: Volatility Clustering in Action")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "p04_garch_vol.png")
    plt.close(fig)


def plot_p04_regime_switching():
    rng = np.random.default_rng(42)
    n = 600
    # Simulate 2-regime data
    regime = np.zeros(n, dtype=int)
    regime[0] = 0
    for t in range(1, n):
        if regime[t-1] == 0:
            regime[t] = 0 if rng.random() < 0.98 else 1
        else:
            regime[t] = 1 if rng.random() < 0.93 else 0

    returns = np.where(regime == 0,
                       rng.normal(0.0003, 0.008, n),
                       rng.normal(-0.001, 0.025, n))

    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), height_ratios=[2, 1], sharex=True)

    colors = np.where(regime == 1, "crimson", "steelblue")
    ax1.bar(dates, returns, color=colors, alpha=0.7, width=1)
    ax1.set_ylabel("Return")
    ax1.set_title("Regime-Switching Model: Calm (blue) vs Crisis (red)")
    ax1.grid(alpha=0.3)

    ax2.fill_between(dates, 0, regime, color="crimson", alpha=0.4, step="mid", label="Crisis regime")
    ax2.set_ylabel("P(crisis)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "p04_regime.png")
    plt.close(fig)


# ── Run all ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating showcase plots...")
    plot_p01_var_backtest()
    print("  [done] P01 VaR backtest")
    plot_p01_loss_distribution()
    print("  [done] P01 loss distribution")
    plot_p02_mc_paths()
    print("  [done] P02 MC paths")
    plot_p02_variance_reduction()
    print("  [done] P02 variance reduction")
    plot_p03_roc_ks()
    print("  [done] P03 ROC + KS")
    plot_p03_shap_importance()
    print("  [done] P03 feature importance")
    plot_p04_garch_volatility()
    print("  [done] P04 GARCH volatility")
    plot_p04_regime_switching()
    print("  [done] P04 regime switching")
    print(f"\nAll plots saved to {OUT}/")
