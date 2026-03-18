"""Generate showcase plots for Project 09 (CVA Counterparty Risk)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "projects" / "09_cva_counterparty_risk" / "src"))
OUT = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "figure.figsize": (7, 3.5),
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

def plot_p09_exposure_profiles():
    from instruments import simulate_rate_paths, InterestRateSwap
    from exposure import compute_exposure_profiles

    paths, times = simulate_rate_paths(r0=0.03, kappa=0.5, theta=0.04, sigma=0.01,
                                        T=5.0, n_steps=20, n_paths=5000, seed=42)
    swap = InterestRateSwap(notional=1e6, fixed_rate=0.04, tenor=5.0, payment_freq=0.25, seed=42)
    mtm = swap.simulate_values(paths, times)
    profiles = compute_exposure_profiles(mtm, times)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(times, profiles["ee"] / 1e3, color="steelblue", linewidth=2, label="Expected Exposure (EE)")
    ax.plot(times, profiles["pfe_975"] / 1e3, color="crimson", linewidth=2, linestyle="--", label="PFE 97.5%")
    ax.fill_between(times, 0, profiles["ee"] / 1e3, alpha=0.15, color="steelblue")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure ($K)")
    ax.set_title("Counterparty Exposure Profile: 5-Year Interest Rate Swap ($1M notional)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "p09_exposure_profiles.png")
    plt.close(fig)
    print(f"  P09: Peak EE = ${profiles['ee'].max()/1e3:.1f}K, Peak PFE = ${profiles['pfe_975'].max()/1e3:.1f}K")

def plot_p09_wrong_way_risk():
    from instruments import simulate_rate_paths, InterestRateSwap
    from exposure import compute_exposure_profiles
    from cva import compute_cva
    from credit import hazard_rate_from_cds

    hr = hazard_rate_from_cds(0.01, 0.40)
    correlations = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    cvas = []

    for corr in correlations:
        rng = np.random.default_rng(42)
        n_paths, n_steps = 5000, 20
        dt = 5.0 / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = 0.03
        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            paths[:, t+1] = paths[:, t] + 0.5 * (0.04 - paths[:, t]) * dt + 0.01 * np.sqrt(dt) * z1

        times = np.linspace(0, 5.0, n_steps + 1)
        swap = InterestRateSwap(notional=1e6, fixed_rate=0.04, tenor=5.0, payment_freq=0.25, seed=42)
        mtm = swap.simulate_values(paths, times)

        # Wrong-way: weight paths where default is more likely (low rates = high exposure AND high default)
        exposure = np.maximum(mtm, 0)
        if corr > 0:
            weights = 1 + corr * (exposure.mean(axis=1) - exposure.mean(axis=1).mean()) / (exposure.mean(axis=1).std() + 1e-10)
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            weighted_ee = np.average(exposure, axis=0, weights=weights)
        else:
            profiles = compute_exposure_profiles(mtm, times)
            weighted_ee = profiles["ee"]

        cva_val = (1 - 0.40) * np.sum(weighted_ee[1:] * (np.exp(-hr * times[:-1]) - np.exp(-hr * times[1:])) * np.exp(-0.03 * times[1:]))
        cvas.append(cva_val)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["steelblue" if c <= 0 else "crimson" for c in correlations]
    bars = ax.bar([f"{c:+.2f}" for c in correlations], [c/1e3 for c in cvas], color=colors, edgecolor="white")
    ax.axhline(cvas[2]/1e3, color="black", linestyle="--", linewidth=0.8, label=f"Zero correlation: ${cvas[2]/1e3:.1f}K")
    ax.set_xlabel("Exposure-Default Correlation")
    ax.set_ylabel("CVA ($K)")
    ax.set_title("Wrong-Way Risk: CVA Increases with Positive Exposure-Default Correlation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "p09_wrong_way_risk.png")
    plt.close(fig)

if __name__ == "__main__":
    print("Generating P09 showcase plots...")
    plot_p09_exposure_profiles()
    print("  [done] P09 exposure profiles")
    plot_p09_wrong_way_risk()
    print("  [done] P09 wrong-way risk")
    print(f"\nAll plots saved to {OUT}/")
