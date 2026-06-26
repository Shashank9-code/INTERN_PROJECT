"""
=============================================================================
benchmark_all.py — Phase 2: Scaled Architecture Benchmarking
=============================================================================

PROJECT STRUCTURE
─────────────────
  Phase 1 — Fundamental Research
      Hand-crafted, monolithic GCN / GAT / GNN scripts preserved in
      `standalone_tutorials/` for educational deep-study.  Each uses
      BCEWithLogitsLoss and manual training loops.

  Phase 2 — Scaled Benchmarking  ← THIS SCRIPT
      Modular pipeline that imports all 8 encoder architectures from
      `models.py` and benchmarks them on identical SNAP Facebook splits
      using **BPR (Bayesian Personalized Ranking) Loss** — a ranking-
      aware objective specifically designed for recommendation.

MODELS (imported from models.py)
────────────────────────────────
  1. GCN           — GCNConv (Kipf & Welling, 2017)
  2. GNN           — GraphConv (vanilla MPNN)
  3. GAT           — GATConv (multi-head attention)
  4. GraphSAGE     — SAGEConv (Hamilton et al., 2017)
  5. GIN           — GINConv (Xu et al., 2019) with 2-layer MLP
  6. TransformerConv — PyG Graph Transformer (Shi et al., 2021)
  7. GATv2         — GATv2Conv (dynamic attention, Brody et al., 2022)
  8. LightGCN      — No transforms, no non-linearities (He et al., 2020)

LOSS FUNCTION
─────────────
  BPR Loss (Bayesian Personalized Ranking):
      L = −mean( log σ(score_pos − score_neg) )

  NO BCEWithLogitsLoss is used anywhere in this pipeline.
  BPR explicitly pushes positive-edge dot-product scores above
  negative-edge scores, which is the correct objective for ranking.

EARLY STOPPING
──────────────
  • Monitor: Validation AUC
  • Patience: 25 epochs (no improvement → halt training)
  • Maximum epochs: 200
  • Best model weights are always restored before evaluation.

REGULARIZATION (ICLR 2017 GCN framework)
─────────────────────────────────────────
  • Dropout 0.5 on input and hidden layers
  • L2 weight decay 5e-4 in Adam optimizer

OUTPUTS
───────
  • benchmark_results.csv / .md — consolidated metrics table
  • model_comparison.png        — grouped bar chart (5 metrics)
  • learning_curves_phase2.png  — per-model BPR loss + Val AUC curves
  • all_embeddings.pt           — final-epoch embeddings for t-SNE

Author  : Shashank Prabhakar
Date    : April 2026
"""

# ── Imports ──────────────────────────────────────────────────────────────
import os, time, random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import set_seed, load_facebook_dataset, prepare_splits
from models import MODEL_REGISTRY, LightGCNEncoder
from losses import BPRLoss, split_pos_neg_edges
from evaluate import evaluate_pairwise, evaluate_all


# ══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════
SEED            = 42
HIDDEN          = 128
OUT             = 64
LR              = 0.005
EPOCHS          = 200       # extended from 100 → 200 for full convergence
DROPOUT         = 0.5       # ICLR 2017 GCN recommendation
WEIGHT_DECAY    = 5e-4      # L2 regularisation in Adam
PATIENCE        = 25        # early stopping patience (epochs)
HEADS           = 4         # attention heads for GAT/GATv2/TransformerConv
TOP_K           = 10        # ranking metric cutoff
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All 8 architectures from models.py
MODEL_NAMES = [
    "GCN", "GNN", "GAT", "GraphSAGE",
    "GIN", "TransformerConv", "GATv2", "LightGCN",
]


# ══════════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════
def build_model(name: str, in_ch: int, n_nodes: int):
    """Instantiate an encoder by name from MODEL_REGISTRY."""
    cls = MODEL_REGISTRY[name]

    if name in ("GAT", "GATv2", "TransformerConv"):
        m = cls(in_ch, HIDDEN, OUT, heads=HEADS, dropout=DROPOUT)
    elif name == "LightGCN":
        m = cls(in_ch, HIDDEN, OUT, num_layers=3, dropout=0.0,
                num_nodes=n_nodes)
    else:
        m = cls(in_ch, HIDDEN, OUT, dropout=DROPOUT)

    return m.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP  (BPR Loss — NO BCE anywhere)
# ══════════════════════════════════════════════════════════════════════════
def train_model(name, model, train_data, val_data):
    """
    Train a model with BPR (Bayesian Personalized Ranking) Loss.

    Training step
    ─────────────
    1. Compute node embeddings via the encoder.
    2. Split supervision edges into positive / negative pairs.
    3. Compute dot-product scores for both sets.
    4. Apply BPR: −mean(log σ(score_pos − score_neg)).
    5. Backprop and update weights.

    Early stopping
    ──────────────
    Monitors Validation AUC with patience=25.  If no improvement for
    25 consecutive epochs, training halts and best weights are restored.

    Returns
    -------
    model   : nn.Module — with best weights loaded
    history : dict      — 'train_loss', 'val_auc', 'val_ap', 'stopped_epoch'
    """
    # L2 regularisation via weight_decay (ICLR 2017 GCN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    criterion = BPRLoss()

    best_auc         = 0.0
    best_state       = None
    best_ep          = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_auc":    [],
        "val_ap":     [],
    }

    for ep in range(1, EPOCHS + 1):
        # ── Training step (BPR) ──────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        z = model(train_data.x, train_data.edge_index)

        # Split supervision edges into pos / neg
        pos_edge, neg_edge = split_pos_neg_edges(
            train_data.edge_label_index, train_data.edge_label
        )

        # BPR loss: push positive scores above negative scores
        loss = criterion(z, pos_edge, neg_edge)
        loss.backward()
        optimizer.step()

        # ── Validation ───────────────────────────────────────────────────
        val_auc, val_ap = evaluate_pairwise(model, val_data)

        history["train_loss"].append(loss.item())
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)

        # ── Early stopping on Validation AUC ─────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            best_ep  = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress every 25 epochs
        if ep % 25 == 0 or ep == 1:
            print(f"    Ep {ep:3d}/{EPOCHS}"
                  f"  │  BPR Loss: {loss.item():.4f}"
                  f"  │  Val AUC: {val_auc:.4f}"
                  f"  │  Val AP: {val_ap:.4f}"
                  f"{'  ◀ best' if ep == best_ep else ''}")

        if patience_counter >= PATIENCE:
            print(f"    ⏹  Early stopping at epoch {ep}"
                  f"  (no AUC improvement for {PATIENCE} epochs)")
            break

    history["stopped_epoch"] = ep
    print(f"    ✓  Best Val AUC: {best_auc:.4f}  (epoch {best_ep})")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ══════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS EXTRACTION
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def get_embeddings(model, data):
    """Extract node embeddings from the model."""
    model.eval()
    return model(data.x, data.edge_index).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — Per-Model Learning Curves (BPR Loss + Val AUC)
# ══════════════════════════════════════════════════════════════════════════
def plot_learning_curves(all_histories: dict,
                         save_path: str = "learning_curves_phase2.png"):
    """
    Plot BPR Loss and Validation AUC curves for all models in a 2-panel
    figure.  X-axis scales to the actual number of epochs each model
    trained (up to 200).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Curated colour palette for 8 models
    cmap = plt.cm.get_cmap("tab10", len(all_histories))

    for idx, (name, hist) in enumerate(all_histories.items()):
        color = cmap(idx)
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Left panel: BPR Ranking Loss
        ax1.plot(epochs, hist["train_loss"], label=name,
                 color=color, linewidth=1.5, alpha=0.85)

        # Right panel: Validation AUC
        ax2.plot(epochs, hist["val_auc"], label=name,
                 color=color, linewidth=1.5, alpha=0.85)

        # Mark early stopping point
        stopped = hist.get("stopped_epoch", len(hist["train_loss"]))
        if stopped < EPOCHS:
            ax1.axvline(x=stopped, color=color, linestyle=":",
                        alpha=0.3, linewidth=1)
            ax2.axvline(x=stopped, color=color, linestyle=":",
                        alpha=0.3, linewidth=1)

    # ── Left panel: BPR Loss ─────────────────────────────────────────────
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("BPR Ranking Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Training Loss (BPR) vs. Epochs", fontsize=14,
                  fontweight="bold")
    ax1.set_xlim(1, EPOCHS)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Right panel: Validation AUC ──────────────────────────────────────
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation AUC", fontsize=12, fontweight="bold")
    ax2.set_title("Validation AUC vs. Epochs", fontsize=14,
                  fontweight="bold")
    ax2.set_xlim(1, EPOCHS)
    ax2.set_ylim(0.5, 1.02)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Phase 2 — Learning Curves (BPR Ranking Loss)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  📊  Learning curves saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — Grouped Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def plot_comparison(df, save_path="model_comparison.png"):
    """Grouped bar chart comparing all models on all metrics."""
    metrics = [c for c in df.columns if c != "Model"]
    models  = df["Model"].tolist()
    x = np.arange(len(models))
    w = 0.8 / len(metrics)
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, m in enumerate(metrics):
        off = (i - len(metrics) / 2 + 0.5) * w
        bars = ax.bar(x + off, df[m].values, w, label=m,
                      color=colors[i % len(colors)],
                      edgecolor="white", linewidth=0.5)
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold")

    ax.set_xlabel("Model Architecture", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Phase 2 — Link-Prediction Benchmark (BPR Loss)",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  📊  Bar chart saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN — PHASE 2: SCALED BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Phase 2 Header ───────────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                                                        ║")
    print("  ║   ⚡  PHASE 2 — SCALED ARCHITECTURE BENCHMARKING  ⚡   ║")
    print("  ║                                                        ║")
    print("  ║   Friend Recommendation via Link Prediction            ║")
    print("  ║   8 GNN Architectures  ×  BPR Ranking Loss             ║")
    print("  ║   200 Epochs  ·  Early Stopping (patience=25)          ║")
    print("  ║                                                        ║")
    print("  ║   Phase 1 tutorials preserved in:                      ║")
    print("  ║     └─ standalone_tutorials/                           ║")
    print("  ║                                                        ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Data Loading (shared across all models) ──────────────────────────
    set_seed(SEED)
    data = load_facebook_dataset(root="./data")
    train_data, val_data, test_data = prepare_splits(data, seed=SEED)

    train_data = train_data.to(DEVICE)
    val_data   = val_data.to(DEVICE)
    test_data  = test_data.to(DEVICE)

    in_ch    = train_data.x.size(1)
    n_nodes  = data.num_nodes

    print(f"  Device: {DEVICE}")
    print(f"  Config: epochs={EPOCHS}, patience={PATIENCE}, "
          f"dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}")
    print(f"  Loss:   BPR (Bayesian Personalized Ranking)")
    print(f"  Early Stop: Validation AUC, patience={PATIENCE}")
    print()

    # ── Benchmark Loop ───────────────────────────────────────────────────
    results       = []
    emb_dict      = {}
    all_histories = {}

    for name in MODEL_NAMES:
        print("  ━" * 32)
        print(f"  🚀  [{MODEL_NAMES.index(name)+1}/{len(MODEL_NAMES)}]"
              f"  Training: {name}")
        print("  ━" * 32)

        # Pin seed for fair comparison (identical weight initialisation)
        set_seed(SEED)
        model = build_model(name, in_ch, n_nodes)

        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"    Trainable parameters: {trainable:,}")

        t0 = time.time()
        model, history = train_model(name, model, train_data, val_data)
        dt = time.time() - t0
        print(f"    ⏱  Training time: {dt:.1f}s"
              f"  ({history['stopped_epoch']} epochs)")

        # ── Full evaluation on test set ──────────────────────────────────
        metrics = evaluate_all(model, test_data, train_data.edge_index,
                               K=TOP_K)
        metrics["Model"] = name
        metrics["Epochs"] = history["stopped_epoch"]
        metrics["Params"] = trainable
        metrics["Time/Ep(ms)"] = round((dt / history["stopped_epoch"]) * 1000, 1)
        results.append(metrics)

        print(f"    ✅  AUC={metrics['AUC']:.4f}"
              f"  AP={metrics['AP']:.4f}"
              f"  R@{TOP_K}={metrics[f'Recall@{TOP_K}']:.4f}"
              f"  nDCG@{TOP_K}={metrics[f'nDCG@{TOP_K}']:.4f}"
              f"  MRR={metrics['MRR']:.4f}")

        # Save embeddings + history
        emb_dict[name] = get_embeddings(model, train_data)
        all_histories[name] = history

        # Save model checkpoint
        torch.save(model.state_dict(),
                   f"defense_outputs/best_model_{name.lower().replace(' ', '_')}.pt")

    # ── Build Results DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(results)
    mcols = ["AUC", "AP", f"Recall@{TOP_K}", f"nDCG@{TOP_K}", "MRR"]
    cold_cols = [f"Recall@{TOP_K} (Low Deg)", f"Recall@{TOP_K} (High Deg)"]
    df = df[["Model"] + mcols + cold_cols + ["Epochs", "Params", "Time/Ep(ms)"]]

    # ── Export Results ───────────────────────────────────────────────────
    df.to_csv("defense_outputs/benchmark_results.csv", index=False)
    print(f"\n  💾  CSV saved → defense_outputs/benchmark_results.csv")

    with open("defense_outputs/benchmark_results.md", "w") as f:
        f.write("# Phase 2 — Link-Prediction Benchmark Results\n\n")
        f.write(f"**Dataset:** SNAP Facebook Ego-Network  |  "
                f"**Max Epochs:** {EPOCHS}  |  "
                f"**Loss:** BPR  |  "
                f"**Early Stop:** patience={PATIENCE} on Val AUC  |  "
                f"**Seed:** {SEED}\n\n")
        f.write("> Phase 1 standalone tutorials (GCN, GAT, GNN) are "
                "preserved in `standalone_tutorials/`\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"  💾  Markdown saved → defense_outputs/benchmark_results.md")

    # ── Console Output ───────────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║              PHASE 2 — BENCHMARK RESULTS               ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    print(df.to_string(index=False))
    print()

    # ── Visualizations ───────────────────────────────────────────────────
    plot_comparison(df[["Model"] + mcols], save_path="defense_outputs/model_comparison.png")
    plot_learning_curves(all_histories, save_path="defense_outputs/learning_curves_phase2.png")

    # Accuracy vs Efficiency Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        ax.scatter(row["Time/Ep(ms)"], row[f"nDCG@{TOP_K}"], s=100)
        ax.annotate(row["Model"], (row["Time/Ep(ms)"], row[f"nDCG@{TOP_K}"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Average Time per Epoch (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Test nDCG@{TOP_K}", fontsize=11, fontweight="bold")
    ax.set_title("Accuracy vs. Efficiency Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("defense_outputs/efficiency_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  📊  Efficiency scatter saved → defense_outputs/efficiency_tradeoff.png")

    # ── Save embeddings for t-SNE dashboard ──────────────────────────────
    torch.save(emb_dict, "defense_outputs/all_embeddings.pt")
    print("  💾  Embeddings saved → defense_outputs/all_embeddings.pt")

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  🎉  Phase 2 Benchmark Complete!                       ║")
    print("  ║                                                        ║")
    print("  ║  Next steps:                                           ║")
    print("  ║    python tsne_dashboard.py   ← t-SNE grid             ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
