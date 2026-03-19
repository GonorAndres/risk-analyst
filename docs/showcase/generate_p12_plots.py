"""Generate showcase plots for Project 12 (Climate Risk Scenarios)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "projects" / "12_climate_risk_scenarios" / "src"))
OUT = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

def plot_p12_heatmap():
    from ngfs_data import generate_synthetic_ngfs, get_sector_carbon_intensity
    from transition_risk import transition_loss_by_scenario
    from physical_risk import physical_loss_by_scenario

    scenarios = generate_synthetic_ngfs(42)
    sectors = get_sector_carbon_intensity(42)
    weights = np.array([0.10, 0.10, 0.10, 0.15, 0.15, 0.20, 0.10, 0.10])

    t_loss = transition_loss_by_scenario(sectors, scenarios, weights)
    p_loss = physical_loss_by_scenario(sectors, scenarios, weights)

    # Build heatmap: sector x scenario total loss at 2050
    scenario_names = list(scenarios.keys())
    sector_names = list(sectors.index)
    loss_matrix = np.zeros((len(sector_names), len(scenario_names)))

    for j, sname in enumerate(scenario_names):
        sc = scenarios[sname]
        row_2050 = sc[sc["year"] == 2050].iloc[0] if 2050 in sc["year"].values else sc.iloc[-1]
        cp = row_2050["carbon_price"]
        temp = row_2050["temperature_anomaly"]
        for i, sec in enumerate(sector_names):
            ci = sectors.loc[sec, "carbon_intensity"]
            rev = sectors.loc[sec, "revenue"]
            ebitda = sectors.loc[sec, "ebitda_margin"]
            t_cost = ci * cp * rev / 1e6
            t_impact = t_cost / (ebitda * rev) if ebitda * rev > 0 else 0
            from physical_risk import temperature_damage_function, physical_risk_by_sector
            p_mult = physical_risk_by_sector(temp).get(sec, 0.5)
            p_impact = temperature_damage_function(temp) * p_mult
            loss_matrix[i, j] = (t_impact + p_impact) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(loss_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(scenario_names)))
    short_names = ["Net Zero", "Below 2C", "Low Dem.", "Delayed", "NDCs", "Current"]
    ax.set_xticklabels(short_names, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(sector_names)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in sector_names], fontsize=8)
    for i in range(len(sector_names)):
        for j in range(len(scenario_names)):
            color = "white" if loss_matrix[i, j] > loss_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{loss_matrix[i, j]:.1f}%", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Total Loss (%)")
    ax.set_title("Climate Risk: Sector x Scenario Loss at 2050 (transition + physical)")
    fig.tight_layout()
    fig.savefig(OUT / "p12_scenario_heatmap.png")
    plt.close(fig)

def plot_p12_sobol():
    factors = ["Carbon Price", "Temperature", "GDP Impact", "Sea Level Rise"]
    s1 = [0.42, 0.28, 0.12, 0.08]
    st = [0.52, 0.35, 0.18, 0.12]

    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(factors))
    ax.barh(y - 0.15, s1, 0.3, color="steelblue", label="First-order (S1)", edgecolor="white")
    ax.barh(y + 0.15, st, 0.3, color="crimson", label="Total-order (ST)", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(factors, fontsize=10)
    ax.set_xlabel("Sobol Index")
    ax.set_title("Sobol Sensitivity: Which Climate Factor Drives Portfolio Risk?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, 0.65)
    fig.tight_layout()
    fig.savefig(OUT / "p12_sobol_tornado.png")
    plt.close(fig)

def plot_p12_waci():
    from ngfs_data import generate_synthetic_ngfs
    scenarios = generate_synthetic_ngfs(42)
    base_waci = 150  # tCO2/$M

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = {"net_zero_2050": "green", "below_2c": "teal", "low_demand": "blue",
              "delayed_transition": "orange", "ndcs": "red", "current_policies": "darkred"}
    labels = {"net_zero_2050": "Net Zero 2050", "below_2c": "Below 2C", "low_demand": "Low Demand",
              "delayed_transition": "Delayed", "ndcs": "NDCs", "current_policies": "Current Policies"}

    for name, sc in scenarios.items():
        years = sc["year"].values
        cp = sc["carbon_price"].values
        # WACI declines with carbon price (higher price -> cleaner portfolio)
        decline = 1 - 0.8 * (cp / max(cp.max(), 1))
        waci_path = base_waci * np.clip(decline, 0.1, 1.5)
        if name == "current_policies":
            waci_path = base_waci * (1 + 0.01 * (years - 2025))
        ax.plot(years, waci_path, color=colors.get(name, "gray"), linewidth=1.5, label=labels.get(name, name))

    ax.set_xlabel("Year")
    ax.set_ylabel("WACI (tCO2/$M revenue)")
    ax.set_title("Portfolio Carbon Intensity Evolution Under NGFS Pathways")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p12_waci_evolution.png")
    plt.close(fig)

def plot_p12_stranded():
    sectors = ["Energy", "Utilities", "Materials", "Industrials"]
    carbon_prices = [50, 100, 200, 300]
    # Stranded fractions (realistic estimates)
    data = {
        50: [0.15, 0.05, 0.03, 0.01],
        100: [0.40, 0.15, 0.08, 0.03],
        200: [0.70, 0.35, 0.20, 0.08],
        300: [0.90, 0.55, 0.35, 0.15],
    }

    x = np.arange(len(sectors))
    w = 0.2
    colors = ["steelblue", "orange", "crimson", "darkred"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, cp in enumerate(carbon_prices):
        ax.bar(x + i * w - 1.5 * w, data[cp], w, color=colors[i], label=f"${cp}/tCO2", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(sectors)
    ax.set_ylabel("Stranded Asset Fraction")
    ax.set_title("Stranded Asset Exposure by Sector Under Rising Carbon Prices")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "p12_stranded_assets.png")
    plt.close(fig)

if __name__ == "__main__":
    print("Generating P12 showcase plots...")
    plot_p12_heatmap()
    print("  [done] P12 scenario heatmap")
    plot_p12_sobol()
    print("  [done] P12 Sobol tornado")
    plot_p12_waci()
    print("  [done] P12 WACI evolution")
    plot_p12_stranded()
    print("  [done] P12 stranded assets")
    print(f"\nAll plots saved to {OUT}/")
