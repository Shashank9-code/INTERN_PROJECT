"""
=============================================================================
evaluate.py — Comprehensive Link-Prediction Evaluation
=============================================================================

Provides two levels of evaluation:

  1. **Pairwise metrics** (fast, used every epoch):
     - AUC (Area Under ROC)
     - AP  (Average Precision)

  2. **Top-K ranking metrics** (slower, used for final test evaluation):
     - Recall@K
     - nDCG@K  (Normalized Discounted Cumulative Gain)
     - MRR     (Mean Reciprocal Rank)

     These ranking metrics *mask out training edges* so that the
     evaluation only considers genuinely unseen candidate links.

Author  : Shashank Prabhakar
Date    : April 2026
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ══════════════════════════════════════════════════════════════════════════
#  1.  PAIRWISE EVALUATION  (AUC + AP)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_pairwise(model, data):
    """
    Evaluate the model on a given split using AUC and Average Precision.

    Parameters
    ----------
    model : nn.Module — trained encoder (any architecture from models.py)
    data  : Data      — split data with edge_label_index & edge_label

    Returns
    -------
    auc  : float
    ap   : float
    """
    model.eval()

    z = model(data.x, data.edge_index)

    # Dot-product decode
    src = z[data.edge_label_index[0]]
    dst = z[data.edge_label_index[1]]
    logits = (src * dst).sum(dim=-1)

    probs  = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.cpu().numpy()

    auc = roc_auc_score(labels, probs)
    ap  = average_precision_score(labels, probs)
    return auc, ap


# ══════════════════════════════════════════════════════════════════════════
#  2.  TOP-K RANKING EVALUATION  (Recall@K, nDCG@K, MRR)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_ranking(model, data, train_edge_index, K: int = 10):
    """
    Compute Recall@K, nDCG@K, and MRR for the test set, masking out
    existing training edges so that recommendations are truly novel.

    Strategy
    ────────
    For each *source* node that appears in the test positive edges:
      1. Compute dot-product scores to all other nodes.
      2. Mask out training neighbours and self-links.
      3. Rank candidate nodes by score.
      4. Check which of the test positive targets appear in the top K.

    This is a *per-user* evaluation averaged across all test source nodes.

    Parameters
    ----------
    model            : nn.Module — trained encoder
    data             : Data      — test split (has edge_label_index / edge_label)
    train_edge_index : (2, E)    — training edges to mask
    K                : int       — top-K cutoff (default 10)

    Returns
    recall_at_k : float
    ndcg_at_k   : float
    mrr         : float
    recall_low  : float  (Cold-Start: Bottom 25% degree users)
    recall_high : float  (High-Degree: Top 75% degree users)
    """
    model.eval()

    z = model(data.x, data.edge_index)  # (N, d)
    z = z.cpu()

    num_nodes = z.size(0)

    # ── Build training adjacency set for fast masking ────────────────────
    train_adj = {}
    src_train = train_edge_index[0].cpu().tolist()
    dst_train = train_edge_index[1].cpu().tolist()
    for s, d in zip(src_train, dst_train):
        train_adj.setdefault(s, set()).add(d)

    # ── Identify test positive edges ─────────────────────────────────────
    pos_mask = data.edge_label.cpu() == 1.0
    test_pos_index = data.edge_label_index[:, pos_mask].cpu()

    # Group test positive targets by source node
    test_targets = {}
    for i in range(test_pos_index.size(1)):
        src = test_pos_index[0, i].item()
        dst = test_pos_index[1, i].item()
        test_targets.setdefault(src, set()).add(dst)

    # ── Degree Calculation for Cold-Start ────────────────────────────────
    degrees = np.array([len(train_adj.get(u, set())) for u in range(num_nodes)])
    non_zero_degrees = degrees[degrees > 0]
    p25 = np.percentile(non_zero_degrees, 25) if len(non_zero_degrees) > 0 else 0

    # ── Per-user ranking evaluation ──────────────────────────────────────
    recalls, ndcgs, mrrs = [], [], []
    recalls_low, recalls_high = [], []

    for user, true_targets in test_targets.items():
        if len(true_targets) == 0:
            continue

        # Score all nodes for this user
        user_emb = z[user]  # (d,)
        scores = (z * user_emb).sum(dim=-1)  # (N,)

        # Mask out training neighbours + self
        mask_set = train_adj.get(user, set()) | {user}
        mask_indices = list(mask_set)
        scores[mask_indices] = float("-inf")

        # Top-K candidates
        _, topk_indices = torch.topk(scores, k=min(K, num_nodes))
        topk_list = topk_indices.tolist()

        # ── Recall@K ─────────────────────────────────────────────────────
        hits = len(set(topk_list) & true_targets)
        recall = hits / min(len(true_targets), K)
        recalls.append(recall)

        if degrees[user] <= p25:
            recalls_low.append(recall)
        else:
            recalls_high.append(recall)

        # ── nDCG@K ───────────────────────────────────────────────────────
        dcg = 0.0
        for rank, node in enumerate(topk_list):
            if node in true_targets:
                dcg += 1.0 / np.log2(rank + 2)  # rank is 0-indexed

        # Ideal DCG: all true targets at the top
        ideal_hits = min(len(true_targets), K)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

        # ── MRR ──────────────────────────────────────────────────────────
        rr = 0.0
        for rank, node in enumerate(topk_list):
            if node in true_targets:
                rr = 1.0 / (rank + 1)
                break
        mrrs.append(rr)

    recall_at_k = float(np.mean(recalls)) if recalls else 0.0
    ndcg_at_k   = float(np.mean(ndcgs))   if ndcgs   else 0.0
    mrr         = float(np.mean(mrrs))    if mrrs    else 0.0
    recall_low  = float(np.mean(recalls_low)) if recalls_low else 0.0
    recall_high = float(np.mean(recalls_high)) if recalls_high else 0.0

    return recall_at_k, ndcg_at_k, mrr, recall_low, recall_high


# ══════════════════════════════════════════════════════════════════════════
#  3.  COMBINED EVALUATION  (convenience wrapper)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_all(model, data, train_edge_index, K: int = 10):
    """
    Run both pairwise and ranking evaluation, returning a single dict.

    Returns
    -------
    metrics : dict with keys
        'AUC', 'AP', 'Recall@K', 'nDCG@K', 'MRR'
    """
    auc, ap = evaluate_pairwise(model, data)
    recall_at_k, ndcg_at_k, mrr, recall_low, recall_high = evaluate_ranking(
        model, data, train_edge_index, K=K
    )

    return {
        "AUC":             auc,
        "AP":              ap,
        f"Recall@{K}":     recall_at_k,
        f"nDCG@{K}":       ndcg_at_k,
        "MRR":             mrr,
        f"Recall@{K} (Low Deg)": recall_low,
        f"Recall@{K} (High Deg)": recall_high,
    }
