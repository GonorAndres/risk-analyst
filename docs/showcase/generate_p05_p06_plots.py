"""Generate showcase plots for Projects 05 (EVT) and 06 (Copulas)."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

np.random.seed(42)
plt.rcParams.update({
    "figure.dpi": 200, "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "figure.figsize": (7, 3.5),
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def plot_p05_evt_vs_normal():
    """Bar chart: EVT vs Normal vs t VaR at extreme quantiles."""
    from risk_analyst.models.evt import fit_gpd, evt_var, evt_es

    rng = np.random.default_rng(42)
    losses = rng.pareto(3, size=5000) * 0.01  # heavy-tailed losses

    threshold = np.percentile(losses, 95)
    exceedances = losses[losses > threshold] - threshold
    gpd = fit_gpd(exceedances, threshold)

    alphas = [0.95, 0.99, 0.995, 0.999]
    mu, sigma = losses.mean(), losses.std()
    n_total = len(losses)

    var_normal = [mu + sigma * stats.norm.ppf(a) for a in alphas]
    nu = 5
    var_t = [mu + sigma * stats.t.ppf(a, df=nu) * np.sqrt((nu - 2) / nu) for a in alphas]
    var_evt = [evt_var(gpd, n_total, a) for a in alphas]

    x = np.arange(len(alphas))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - w, var_normal, w, color="steelblue", label="Normal", edgecolor="white")
    ax.bar(x, var_t, w, color="orange", label="Student-t (df=5)", edgecolor="white")
    ax.bar(x + w, var_evt, w, color="crimson", label="EVT (GPD)", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1%}" for a in alphas])
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Value-at-Risk")
    ax.set_title("VaR Comparison: Normal vs Student-t vs EVT at Extreme Quantiles")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "p05_evt_vs_normal.png")
    plt.close(fig)
    print(f"  P05: Normal 99.9% VaR={var_normal[-1]:.4f}, EVT={var_evt[-1]:.4f} "
          f"(underestimation: {(1 - var_normal[-1]/var_evt[-1])*100:.0f}%)")


def plot_p05_gpd_tail():
    """GPD fit to the tail with exceedance distribution."""
    rng = np.random.default_rng(42)
    losses = rng.pareto(3, size=5000) * 0.01

    threshold = np.percentile(losses, 95)
    exceedances = losses[losses > threshold] - threshold

    xi, loc, scale = stats.genpareto.fit(exceedances, floc=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Left: exceedance histogram + GPD fit
    x_range = np.linspace(0, exceedances.max(), 200)
    ax1.hist(exceedances, bins=40, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.plot(x_range, stats.genpareto.pdf(x_range, xi, loc=0, scale=scale),
             color="crimson", linewidth=2, label=f"GPD (xi={xi:.3f})")
    ax1.set_xlabel("Excess Loss Above Threshold")
    ax1.set_ylabel("Density")
    ax1.set_title("GPD Fit to Tail Exceedances")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Right: QQ plot
    theoretical = stats.genpareto.ppf(
        np.linspace(0.01, 0.99, len(exceedances)), xi, loc=0, scale=scale
    )
    empirical = np.sort(exceedances)[:len(theoretical)]
    ax2.scatter(theoretical, empirical, s=8, alpha=0.5, color="steelblue")
    lim = max(theoretical.max(), empirical.max())
    ax2.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect fit")
    ax2.set_xlabel("Theoretical Quantiles (GPD)")
    ax2.set_ylabel("Empirical Quantiles")
    ax2.set_title(f"QQ Plot (shape xi = {xi:.3f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "p05_gpd_tail.png")
    plt.close(fig)


def plot_p06_copula_scatter():
    """Scatter plots showing different copula dependence structures."""
    from risk_analyst.models.copula import copula_sample

    n = 2000
    rho = 0.6

    # Gaussian copula
    gauss_params = {"family": "gaussian", "corr_matrix": np.array([[1, rho], [rho, 1]])}
    u_gauss = copula_sample(gauss_params, n, seed=42)

    # t-copula (df=3 for visible tail dependence)
    t_params = {"family": "t", "corr_matrix": np.array([[1, rho], [rho, 1]]), "df": 3}
    u_t = copula_sample(t_params, n, seed=42)

    # Clayton
    theta_c = 2 * 0.5 / (1 - 0.5)  # tau=0.5
    clay_params = {"family": "clayton", "theta": theta_c}
    u_clay = copula_sample(clay_params, n, seed=42)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.8))

    for ax, data, title, color in [
        (ax1, u_gauss, "Gaussian", "steelblue"),
        (ax2, u_t, "Student-t (df=3)", "orange"),
        (ax3, u_clay, "Clayton", "crimson"),
    ]:
        ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.3, color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    ax1.set_ylabel("U2")
    ax2.set_xlabel("U1")
    fig.suptitle("Copula Dependence Structures (same correlation, different tails)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "p06_copula_scatter.png")
    plt.close(fig)


def plot_p06_var_by_copula():
    """Bar chart of portfolio VaR under different copula assumptions."""
    from risk_analyst.models.copula import (
        copula_sample, gaussian_copula_fit, t_copula_fit,
        clayton_copula_fit, pit_transform,
    )

    rng = np.random.default_rng(42)
    n = 5000

    # Simulate 3-asset returns with t-copula dependence
    rho = np.array([[1, 0.6, 0.3], [0.6, 1, 0.5], [0.3, 0.5, 1]])
    L = np.linalg.cholesky(rho)
    df_true = 4
    chi2 = rng.chisquare(df_true, size=n)
    z = rng.standard_normal((n, 3)) @ L.T
    t_samples = z / np.sqrt(chi2[:, None] / df_true)
    returns = t_samples * 0.01  # scale to realistic returns

    weights = np.array([0.4, 0.35, 0.25])
    portfolio_losses = -(returns @ weights)

    # PIT
    u_data = pit_transform(returns[:, :2])  # bivariate for Archimedean

    # Compute VaR under each assumption
    families = ["Normal", "Gaussian Copula", "t-Copula", "Clayton Copula"]
    alphas = [0.95, 0.99]

    var_results = {}
    for alpha in alphas:
        # Normal assumption
        mu, sig = portfolio_losses.mean(), portfolio_losses.std()
        var_normal = mu + sig * stats.norm.ppf(alpha)

        # Gaussian copula (same as normal for Gaussian marginals, but conceptually different)
        var_gauss = np.percentile(portfolio_losses, alpha * 100)

        # t-copula (heavier tails)
        t_params = {"family": "t", "corr_matrix": rho[:2, :2], "df": 4}
        u_sim = copula_sample(t_params, 10000, seed=42)
        # Map back through inverse normal
        sim_returns = stats.norm.ppf(u_sim) * returns[:, :2].std(axis=0)
        sim_losses = -(sim_returns @ weights[:2] / weights[:2].sum())
        var_t = np.percentile(sim_losses, alpha * 100)

        # Clayton (lower tail)
        clay_params = {"family": "clayton", "theta": 2.0}
        u_clay = copula_sample(clay_params, 10000, seed=42)
        sim_returns_c = stats.norm.ppf(u_clay) * returns[:, :2].std(axis=0)
        sim_losses_c = -(sim_returns_c @ weights[:2] / weights[:2].sum())
        var_clay = np.percentile(sim_losses_c, alpha * 100)

        var_results[alpha] = [var_normal, var_gauss, var_t, var_clay]

    x = np.arange(len(families))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - w/2, var_results[0.95], w, color="steelblue", label="95% VaR", edgecolor="white")
    ax.bar(x + w/2, var_results[0.99], w, color="crimson", label="99% VaR", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel("Value-at-Risk")
    ax.set_title("Portfolio VaR Under Different Dependence Assumptions")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "p06_var_by_copula.png")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating P05/P06 showcase plots...")
    plot_p05_evt_vs_normal()
    print("  [done] P05 EVT vs Normal VaR")
    plot_p05_gpd_tail()
    print("  [done] P05 GPD tail fit")
    plot_p06_copula_scatter()
    print("  [done] P06 copula scatter")
    plot_p06_var_by_copula()
    print("  [done] P06 VaR by copula")
    print(f"\nAll plots saved to {OUT}/")
