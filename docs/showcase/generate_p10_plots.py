"""Generate showcase plots for Project 10 (GNN Credit Contagion)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "projects" / "10_gnn_credit_contagion" / "src"))
OUT = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

def plot_p10_network_cascade():
    from network import generate_financial_network
    from contagion import systemic_importance, simulate_cascade

    net = generate_financial_network(n_nodes=40, n_edges=120, seed=42)
    dr = systemic_importance(net["liabilities"], net["assets"])

    # Cascade from most systemic node
    top_node = int(np.argmax(dr))
    cascade = simulate_cascade(net["liabilities"], net["assets"], [top_node], 0.8)

    # Simple force-directed layout
    rng = np.random.default_rng(42)
    pos = rng.uniform(-1, 1, (40, 2))
    A = (net["adjacency"] > 0).astype(float)
    for _ in range(100):
        for i in range(40):
            force = np.zeros(2)
            for j in range(40):
                diff = pos[i] - pos[j]
                dist = max(np.linalg.norm(diff), 0.01)
                force += diff / (dist ** 2) * 0.01  # repulsion
                if A[i, j] > 0 or A[j, i] > 0:
                    force -= diff * 0.05  # attraction
            pos[i] += np.clip(force, -0.1, 0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Left: network colored by DebtRank
    sizes = 30 + dr * 300
    sc = ax1.scatter(pos[:, 0], pos[:, 1], c=dr, cmap="RdYlGn_r", s=sizes, edgecolors="black", linewidth=0.5, zorder=3)
    for i in range(40):
        for j in range(40):
            if A[i, j] > 0:
                ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "gray", alpha=0.15, linewidth=0.5)
    ax1.scatter(pos[top_node, 0], pos[top_node, 1], s=200, facecolors="none", edgecolors="red", linewidth=2, zorder=4)
    ax1.set_title("Network (color = DebtRank)")
    ax1.set_axis_off()
    plt.colorbar(sc, ax=ax1, shrink=0.7, label="DebtRank")

    # Right: cascade size distribution
    cascade_sizes = []
    for node in range(40):
        c = simulate_cascade(net["liabilities"], net["assets"], [node], 0.8)
        cascade_sizes.append(c["n_defaults"])

    colors = ["crimson" if s > np.median(cascade_sizes) else "steelblue" for s in cascade_sizes]
    ax2.bar(range(40), sorted(cascade_sizes, reverse=True), color=sorted(colors, key=lambda c: c == "steelblue"), edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Node (sorted by cascade size)")
    ax2.set_ylabel("Defaults triggered")
    ax2.set_title("Cascade Size per Shocked Node")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "p10_network_cascade.png")
    plt.close(fig)
    print(f"  P10: Top DebtRank node={top_node}, cascade defaults={cascade['n_defaults']}/{40}")

def plot_p10_debtrank_vs_centrality():
    from network import generate_financial_network, compute_centrality
    from contagion import systemic_importance

    net = generate_financial_network(n_nodes=40, n_edges=120, seed=42)
    dr = systemic_importance(net["liabilities"], net["assets"])
    cent = compute_centrality(net["adjacency"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sc = ax.scatter(cent["degree_centrality"], dr, c=net["node_features"][:, 0],
                    cmap="RdYlGn", s=50, edgecolors="black", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Degree Centrality")
    ax.set_ylabel("DebtRank (systemic importance)")
    ax.set_title("DebtRank vs Degree Centrality (color = capital ratio)")
    plt.colorbar(sc, ax=ax, shrink=0.7, label="Capital Ratio")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "p10_debtrank_centrality.png")
    plt.close(fig)

if __name__ == "__main__":
    print("Generating P10 showcase plots...")
    plot_p10_network_cascade()
    print("  [done] P10 network + cascade")
    plot_p10_debtrank_vs_centrality()
    print("  [done] P10 DebtRank vs centrality")
    print(f"\nAll plots saved to {OUT}/")
