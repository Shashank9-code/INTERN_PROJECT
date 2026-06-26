"""
tsne_dashboard.py — 2×4 Grid t-SNE Visualization for All 8 Models

Generates a side-by-side comparison of final-epoch node embeddings,
coloured by Louvain community, for all 8 GNN architectures.

Can be run standalone (after benchmark_all.py saves all_embeddings.pt)
or imported and called directly.

Author  : Shashank Prabhakar
Date    : April 2026
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import community.community_louvain as louvain
except ImportError:
    louvain = None

from torch_geometric.utils import to_networkx
from data_utils import set_seed, load_facebook_dataset

SEED = 42


def compute_louvain_labels(data):
    """Detect Louvain communities and return label array."""
    if louvain is None or data is None:
        print("  ⚠️   Louvain not available — using rainbow fallback")
        return np.arange(data.num_nodes), 0

    G_nx = to_networkx(data, to_undirected=True)
    partition = louvain.best_partition(G_nx)
    labels = np.array([partition[i] for i in range(data.num_nodes)])
    n_comm = labels.max() + 1
    print(f"  ✅  Louvain communities: {n_comm}")
    return labels, n_comm


def plot_tsne_dashboard(embeddings_dict, data,
                        save_path="tsne_dashboard.png"):
    """
    Generate a 2×4 grid of t-SNE scatter plots for all 8 models.

    Parameters
    ----------
    embeddings_dict : dict[str, np.ndarray]
        Mapping of model name → (N, d) embedding array.
    data : torch_geometric.data.Data
        Original graph (for Louvain community detection).
    save_path : str
        Output image path.
    """
    print("\n  🔄  Generating t-SNE dashboard …")

    cluster_labels, n_comm = compute_louvain_labels(data)

    model_names = list(embeddings_dict.keys())
    nrows, ncols = 2, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
    axes_flat = axes.flatten()

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=15,
                max_iter=1000)

    for idx, name in enumerate(model_names):
        ax = axes_flat[idx]
        emb = embeddings_dict[name]
        print(f"    t-SNE for {name} …")

        proj = tsne.fit_transform(emb)
        ax.scatter(proj[:, 0], proj[:, 1],
                   c=cluster_labels, cmap="tab20",
                   s=6, alpha=0.75)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(len(model_names), nrows * ncols):
        axes_flat[idx].set_visible(False)

    suffix = (f" (coloured by {n_comm} Louvain communities)"
              if n_comm > 0 else "")
    fig.suptitle(
        f"t-SNE Node Embeddings — Final Epoch (100){suffix}",
        fontsize=17, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  📊  t-SNE dashboard saved → {save_path}")


def plot_individual_tsne(embeddings_dict, data,
                         output_dir="tsne_individual"):
    """
    Save individual t-SNE plots for each model as separate images.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    cluster_labels, n_comm = compute_louvain_labels(data)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=15,
                max_iter=1000)

    for name, emb in embeddings_dict.items():
        print(f"    Individual t-SNE: {name} …")
        proj = tsne.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(8, 7))
        scatter = ax.scatter(proj[:, 0], proj[:, 1],
                             c=cluster_labels, cmap="tab20",
                             s=8, alpha=0.8)
        ax.set_title(f"{name} — Epoch 100 Embeddings",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE Dim 1", fontsize=11)
        ax.set_ylabel("t-SNE Dim 2", fontsize=11)
        ax.grid(True, alpha=0.2)

        if n_comm > 0:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("Community ID", fontsize=10)

        save_path = os.path.join(output_dir, f"tsne_{name.lower()}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      Saved → {save_path}")


if __name__ == "__main__":
    print("═" * 60)
    print("  t-SNE Dashboard Generator")
    print("═" * 60)

    # Load embeddings saved by benchmark_all.py
    emb_path = "all_embeddings.pt"
    try:
        emb_dict = torch.load(emb_path, weights_only=False)
        print(f"  Loaded embeddings from {emb_path}")
        print(f"  Models: {list(emb_dict.keys())}")
    except FileNotFoundError:
        print(f"  ❌  {emb_path} not found.")
        print("  Run benchmark_all.py first to generate embeddings.")
        exit(1)

    # Load original data for Louvain
    set_seed(SEED)
    data = load_facebook_dataset(root="./data")

    # Grid dashboard
    plot_tsne_dashboard(emb_dict, data, save_path="tsne_dashboard.png")

    # Individual plots
    plot_individual_tsne(emb_dict, data, output_dir="tsne_individual")

    print("\n  🎉  All t-SNE visualizations generated!\n")
