"""Generate showcase plots for Project 07 -- Stress Testing Framework.

Produces:
    figures/p07_scenario_comparison.png  -- bar chart of losses across scenarios
    figures/p07_factor_sensitivity.png   -- horizontal bar chart of factor betas
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project src/ is importable
_SRC = str(Path(__file__).resolve().parent.parent.parent / "projects" / "07_stress_testing_framework" / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transmission import MacroTransmissionModel
from scenarios import get_dfast_scenarios, get_historical_scenarios

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic data (same as tests)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 40
macro_factors = pd.DataFrame({
    "gdp_growth": rng.normal(0.02, 0.01, n),
    "unemployment": rng.normal(0.05, 0.01, n),
    "equity_index": rng.normal(0.02, 0.05, n),
    "interest_rate_10y": rng.normal(0.03, 0.005, n),
    "credit_spread": rng.normal(0.02, 0.005, n),
    "house_price_index": rng.normal(0.01, 0.02, n),
})
betas_true = np.array([0.5, -0.3, 0.8, -0.2, -0.6, 0.3])
portfolio_returns = (macro_factors.values @ betas_true) + rng.normal(0, 0.01, n)

# Fit model
model = MacroTransmissionModel()
model.fit(portfolio_returns, macro_factors)

# ---------------------------------------------------------------------------
# Compute losses for each scenario
# ---------------------------------------------------------------------------
# Historical
historical = get_historical_scenarios()
hist_losses = {name: model.predict_loss(shocks) for name, shocks in historical.items()}

# DFAST severely adverse -- use mean of factor values as single-period shock
dfast = get_dfast_scenarios()
sev_adv_shocks = dfast["severely_adverse"].mean().to_dict()
sev_adv_loss = model.predict_loss(sev_adv_shocks)

# Combine
all_scenarios = {
    "GFC 2008": hist_losses["gfc_2008"],
    "COVID 2020": hist_losses["covid_2020"],
    "SVB 2023": hist_losses["svb_2023"],
    "DFAST Sev. Adverse": sev_adv_loss,
}

# ---------------------------------------------------------------------------
# Plot 1: Scenario comparison bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
names = list(all_scenarios.keys())
losses = [all_scenarios[k] for k in names]
colors = ["#c0392b", "#e67e22", "#2980b9", "#8e44ad"]

bars = ax.bar(names, losses, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
for bar, loss in zip(bars, losses):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{loss:.2%}",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

ax.set_ylabel("Predicted Portfolio Loss", fontsize=12)
ax.set_title("Scenario Comparison: Portfolio Loss Under Stress", fontsize=13, fontweight="bold")
ax.axhline(y=0, color="black", linewidth=0.6)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=min(0, min(losses) - 0.02))
plt.tight_layout()

out1 = FIGURES_DIR / "p07_scenario_comparison.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Factor sensitivity horizontal bar chart
# ---------------------------------------------------------------------------
sens = model.sensitivity_table()
sens_sorted = sens.reindex(sens["beta"].abs().sort_values().index)

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#27ae60" if b > 0 else "#c0392b" for b in sens_sorted["beta"].values]
ax.barh(
    sens_sorted["factor"].values,
    sens_sorted["beta"].values,
    color=bar_colors, edgecolor="black", linewidth=0.5, height=0.55,
)

for i, (factor, beta) in enumerate(zip(sens_sorted["factor"], sens_sorted["beta"])):
    offset = 0.02 if beta >= 0 else -0.02
    ha = "left" if beta >= 0 else "right"
    ax.text(beta + offset, i, f"{beta:.3f}", va="center", ha=ha, fontsize=10, fontweight="bold")

ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Estimated Beta (Sensitivity)", fontsize=12)
ax.set_title("Factor Sensitivity: Which Macro Driver Matters Most?", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()

out2 = FIGURES_DIR / "p07_factor_sensitivity.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved {out2}")

print("Done. All P07 plots generated.")
