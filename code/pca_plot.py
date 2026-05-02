import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("enron_pca_full.csv")

# (Optional) color by influence score if available
use_color = False
try:
    inf = pd.read_csv("all_nodes_ranked.csv")  # must contain 'node','influence_score'
    df = df.merge(inf[["node", "influence_score"]], on="node")
    use_color = True
except Exception:
    pass

# 2. CREATE SUBPLOTS
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

pairs = [
    ("PC1", "PC2"),
    ("PC2", "PC3"),
    ("PC1", "PC3"),
]

for ax, (x, y) in zip(axes, pairs):
    if use_color:
        sc = ax.scatter(df[x], df[y], c=df["influence_score"],
                        cmap="viridis", s=5, alpha=0.6)
    else:
        sc = ax.scatter(df[x], df[y], s=5, alpha=0.6)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    ax.grid(True)

# Colorbar (only if colored)
if use_color:
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Influence Score")

# 3. SAVE / SHOW
plt.savefig("pca_pairwise_plots.png", dpi=300)
plt.show()