# Phase 1 — Fundamental Research (Standalone Tutorials)

This directory contains the **original, self-contained** implementations of
individual GNN architectures for friend recommendation via link prediction.

Each script is a complete, monolithic pipeline — data loading, feature
engineering, model definition, training, evaluation, and visualization — all
in a single file.  They are preserved here for **educational and deep-study
purposes**.

## Scripts

| File | Architecture | Key Concept |
|------|-------------|-------------|
| `friend_recommendation_gcn.py` | GCN (Kipf & Welling, 2017) | Symmetric-normalised adjacency, spectral convolution |
| `friend_recommendation_gat.py` | GAT (Veličković et al., 2018) | Multi-head attention over neighbours |
| `friend_recommendation_gnn.py` | GraphConv (Vanilla MPNN) | Basic message passing without normalisation |
| `ablation_study_layers.py` | GCN (1L / 2L / 5L) | Over-smoothing analysis with variable depth |

## How to Run

Each script runs independently from the project root:

```bash
cd /path/to/MINOR_PROJECT
python standalone_tutorials/friend_recommendation_gcn.py
```

> **Note:** These scripts use `BCEWithLogitsLoss` (classification-based).
> The modular Phase 2 pipeline in the root directory uses `BPR Loss`
> (ranking-based), which is more appropriate for recommendation.
