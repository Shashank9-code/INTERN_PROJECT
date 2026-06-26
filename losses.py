"""
=============================================================================
losses.py — BPR (Bayesian Personalized Ranking) Loss for Link Prediction
=============================================================================

Replaces the standard BCEWithLogitsLoss with a ranking-aware loss that
explicitly pushes the dot-product score of positive edges higher than
negative edges by a margin.

Two loss variants are provided:
  • BPRLoss   — soft-margin pairwise ranking (Rendle et al., 2009)
  • MarginLoss — hard-margin contrastive variant

Author  : Shashank Prabhakar
Date    : April 2026
"""

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════
# 1.  BPR Loss  (Bayesian Personalized Ranking)
# ══════════════════════════════════════════════════════════════════════════
class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss (Rendle et al., 2009).

    Given node embeddings `z`, positive edges, and negative edges, compute:

        L = −(1/|P|) Σ_{(u,v⁺,v⁻)} log σ( z_u·z_{v⁺} − z_u·z_{v⁻} )

    Intuition
    ─────────
    For every positive edge (u, v⁺), we sample a corresponding negative
    edge (u, v⁻) and push the positive score *above* the negative one.
    The sigmoid-log formulation provides a smooth, differentiable loss
    surface that naturally scales gradients when the margin is small.

    Parameters
    ----------
    z              : (N, d)  — node embeddings
    pos_edge_index : (2, P)  — source / destination of positive edges
    neg_edge_index : (2, P)  — source / destination of negative edges

    Returns
    -------
    loss : scalar tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, z, pos_edge_index, neg_edge_index):
        # Positive scores: dot product for each positive edge
        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)

        # Negative scores: dot product for each negative edge
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

        # BPR loss: −mean( log σ(pos − neg) )
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()

        return loss


# ══════════════════════════════════════════════════════════════════════════
# 2.  Margin-Based Contrastive Loss
# ══════════════════════════════════════════════════════════════════════════
class MarginLoss(nn.Module):
    """
    Margin-based contrastive ranking loss.

        L = (1/|P|) Σ max(0,  margin − (score_pos − score_neg))

    This is a harder variant that enforces a fixed gap between positive
    and negative scores.  Once the gap exceeds `margin`, there is zero
    gradient — no further separation is learned.

    Parameters
    ----------
    margin : float — required minimum gap between pos and neg scores
                     (default: 1.0).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, pos_edge_index, neg_edge_index):
        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

        loss = torch.clamp(self.margin - (pos_scores - neg_scores), min=0).mean()
        return loss


# ══════════════════════════════════════════════════════════════════════════
#  HELPER — Split edge_label_index into pos / neg using edge_label
# ══════════════════════════════════════════════════════════════════════════
def split_pos_neg_edges(edge_label_index, edge_label):
    """
    Given the combined supervision edges and their binary labels, return
    separate positive and negative edge_index tensors.

    Parameters
    ----------
    edge_label_index : (2, M) — combined pos + neg edge pairs
    edge_label       : (M,)   — 1.0 for positive, 0.0 for negative

    Returns
    -------
    pos_edge_index : (2, P)
    neg_edge_index : (2, Q)
    """
    pos_mask = edge_label == 1.0
    neg_mask = edge_label == 0.0
    pos_edge_index = edge_label_index[:, pos_mask]
    neg_edge_index = edge_label_index[:, neg_mask]
    return pos_edge_index, neg_edge_index
