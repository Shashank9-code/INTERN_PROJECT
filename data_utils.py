"""
=============================================================================
data_utils.py — Shared Data Loading & Edge Splitting
=============================================================================

Centralises dataset download, feature engineering, and edge splitting so
that every model in the benchmark starts from the *exact same* data.

Author  : Shashank Prabhakar
Date    : April 2026
"""

import random
import numpy as np
import torch
import networkx as nx

from torch_geometric.datasets import SNAPDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx


# ══════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    """Pin all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING  +  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
def load_facebook_dataset(root: str = "./data"):
    """
    Download the SNAP Facebook ego-network and engineer structural features.

    Features appended:
      • Node Degree (normalised to [0, 1])
      • PageRank     (normalised to [0, 1])

    Returns
    -------
    data : torch_geometric.data.Data
    """
    print("=" * 60)
    print("  Loading SNAP Facebook dataset")
    print("=" * 60)

    dataset = SNAPDataset(root=root, name="ego-Facebook")
    data = dataset[0]

    print(f"  Nodes          : {data.num_nodes}")
    print(f"  Edges          : {data.num_edges}  (directed count)")
    print(f"  Node features  : {data.num_node_features}")
    print()

    # ── Feature Engineering ──────────────────────────────────────────────
    print("  🔧  Engineering structural features …")

    G = to_networkx(data, to_undirected=True)

    # Node Degree (normalised)
    degree_dict = dict(G.degree())
    degree_vals = np.array(
        [degree_dict[i] for i in range(data.num_nodes)], dtype=np.float32
    )
    deg_max = degree_vals.max() if degree_vals.max() > 0 else 1.0
    degree_vals /= deg_max

    # PageRank (normalised)
    pr_dict = nx.pagerank(G, alpha=0.85)
    pr_vals = np.array(
        [pr_dict[i] for i in range(data.num_nodes)], dtype=np.float32
    )
    pr_max = pr_vals.max() if pr_vals.max() > 0 else 1.0
    pr_vals /= pr_max

    identity = torch.eye(data.num_nodes, dtype=torch.float)
    degree_tensor = torch.tensor(degree_vals).unsqueeze(1)
    pr_tensor = torch.tensor(pr_vals).unsqueeze(1)

    data.x = torch.cat([identity, degree_tensor, pr_tensor], dim=1)

    print(f"  ✅  Features: identity({data.num_nodes}) + degree(1) + pagerank(1)")
    print(f"      Final feature dim: {data.x.size(1)}")
    print()

    return data


# ══════════════════════════════════════════════════════════════════════════
#  EDGE SPLITTING  &  NEGATIVE SAMPLING
# ══════════════════════════════════════════════════════════════════════════
def prepare_splits(data, seed: int = 42):
    """
    Split edges into train / val / test with negative sampling.

    Uses a fixed seed inside the splitter to guarantee identical splits
    across all model runs.

    Returns
    -------
    train_data, val_data, test_data : Data objects
    """
    print("=" * 60)
    print("  Splitting edges & generating negative samples")
    print("=" * 60)

    # Pin seed right before splitting so splits are deterministic
    set_seed(seed)

    splitter = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        split_labels=False,
    )

    train_data, val_data, test_data = splitter(data)

    for name, split in [("Train", train_data),
                         ("Val",   val_data),
                         ("Test",  test_data)]:
        num_pos = int(split.edge_label.sum().item())
        num_neg = len(split.edge_label) - num_pos
        print(f"  {name:5s}  |  msg edges: {split.edge_index.size(1):>7,}"
              f"  |  pos: {num_pos:>6,}  neg: {num_neg:>6,}")

    print()
    return train_data, val_data, test_data
