"""
=============================================================================
models.py — All GNN Encoder Architectures for Link Prediction
=============================================================================

Contains 8 encoder classes, each producing node embeddings via a uniform
forward(x, edge_index) → (N, out_channels) interface:

  1. GCNEncoder        — GCNConv  (Kipf & Welling, 2017)
  2. GraphConvEncoder  — GraphConv (vanilla MPNN, no symmetric norm)
  3. GATEncoder        — GATConv  (multi-head attention)
  4. GraphSAGEEncoder  — SAGEConv (Hamilton et al., 2017)
  5. GINEncoder        — GINConv  (Xu et al., 2019) with 2-layer MLP
  6. TransformerEncoder— TransformerConv (Shi et al., 2021)
  7. GATv2Encoder      — GATv2Conv (Brody et al., 2022)
  8. LightGCNEncoder   — LightGCN  (He et al., 2020)

Regularization (ICLR 2017 GCN framework)
─────────────────────────────────────────
  • Dropout 0.5 on input features AND between hidden layers.
  • Residual connections for models with num_layers > 2:
        H^{(l+1)} = σ(Conv(H^{(l)})) + H^{(l)}
    This preserves gradient flow and prevents performance degradation
    in deeper networks.

Author  : Shashank Prabhakar
Date    : April 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GraphConv,
    GATConv,
    SAGEConv,
    GINConv,
    TransformerConv as PyGTransformerConv,
    GATv2Conv,
)


# ══════════════════════════════════════════════════════════════════════════
# 1.  GCN  (Graph Convolutional Network)
# ══════════════════════════════════════════════════════════════════════════
class GCNEncoder(nn.Module):
    """
    Variable-depth GCN with symmetric-normalised adjacency aggregation.

    h_v^{(l+1)} = σ( Σ_{u ∈ N(v)} (1/√(d_u·d_v)) · W^{(l)} · h_u^{(l)} )

    When num_layers > 2, residual connections are added to intermediate
    hidden layers:  H^{(l+1)} = ReLU(GCNConv(H^{(l)})) + H^{(l)}
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, dropout: float = 0.5,
                 num_layers: int = 2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = num_layers > 2

        self.convs = nn.ModuleList()
        # Layer 1: in → hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Intermediate layers: hidden → hidden (with residual)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Final layer: hidden → out
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Input dropout (ICLR 2017 GCN)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # not the final layer
                x = F.relu(x)
                # Residual connection for intermediate hidden layers
                if self.use_residual and i > 0:
                    x = x + x_prev
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 2.  GraphConv  (Vanilla MPNN)
# ══════════════════════════════════════════════════════════════════════════
class GraphConvEncoder(nn.Module):
    """
    Variable-depth vanilla MPNN (no symmetric normalisation).

    h_v' = W_1 h_v  +  W_2 · mean_{u ∈ N(v)} h_u

    Residual connections for num_layers > 2.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, dropout: float = 0.5,
                 num_layers: int = 2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = num_layers > 2

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.convs.append(GraphConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.use_residual and i > 0:
                    x = x + x_prev
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 3.  GAT  (Graph Attention Network — v1)
# ══════════════════════════════════════════════════════════════════════════
class GATEncoder(nn.Module):
    """
    Variable-depth GAT with multi-head attention (Veličković et al., 2018).

    Layer 1 : GATConv(in, hidden, heads, concat=True)  → dim = hidden×heads
    Hidden  : GATConv(hidden×heads, hidden, heads, concat=True) + residual
    Final   : GATConv(hidden×heads, out, heads=1, concat=False)

    Residual connections for num_layers > 2.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, heads: int = 4,
                 dropout: float = 0.5, num_layers: int = 2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = num_layers > 2

        self.convs = nn.ModuleList()
        # Layer 1: in → hidden (concat → hidden*heads)
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads=heads, concat=True, dropout=dropout))
        # Intermediate layers: hidden*heads → hidden (concat → hidden*heads)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, concat=True, dropout=dropout))
        # Final layer: hidden*heads → out
        self.convs.append(GATConv(hidden_channels * heads, out_channels,
                                  heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                # Residual: intermediate hidden layers have matching dims
                if self.use_residual and i > 0:
                    x = x + x_prev
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 4.  GraphSAGE  (Sample-and-Aggregate)
# ══════════════════════════════════════════════════════════════════════════
class GraphSAGEEncoder(nn.Module):
    """
    Variable-depth GraphSAGE using SAGEConv (Hamilton et al., 2017).

    Aggregates sampled neighbours independently of graph size.
    Residual connections for num_layers > 2.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, dropout: float = 0.5,
                 num_layers: int = 2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = num_layers > 2

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.use_residual and i > 0:
                    x = x + x_prev
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 5.  GIN  (Graph Isomorphism Network)
# ══════════════════════════════════════════════════════════════════════════
class GINEncoder(nn.Module):
    """
    Variable-depth GIN with 2-layer MLP per GINConv (Xu et al., 2019).

    GINConv applies:  h_v' = MLP( (1 + ε) · h_v  +  Σ_{u ∈ N(v)} h_u )

    Residual connections for num_layers > 2.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, dropout: float = 0.5,
                 num_layers: int = 2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = num_layers > 2

        self.convs = nn.ModuleList()

        # Layer 1: in → hidden
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp1, train_eps=True))

        # Intermediate layers: hidden → hidden
        for _ in range(num_layers - 2):
            mlp_mid = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp_mid, train_eps=True))

        # Final layer: hidden → out
        mlp_last = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.convs.append(GINConv(mlp_last, train_eps=True))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.use_residual and i > 0:
                    x = x + x_prev
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 6.  TransformerConv  (Graph Transformer)
# ══════════════════════════════════════════════════════════════════════════
class TransformerEncoder(nn.Module):
    """
    Variable-depth Graph Transformer using PyG's TransformerConv.

    Residual connections for num_layers > 2.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, heads: int = 4,
                 dropout: float = 0.5, num_layers: int = 2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = num_layers > 2

        self.convs = nn.ModuleList()
        # Layer 1
        self.convs.append(PyGTransformerConv(
            in_channels, hidden_channels // heads,
            heads=heads, concat=True, dropout=dropout))
        # Intermediate layers: hidden → hidden
        for _ in range(num_layers - 2):
            self.convs.append(PyGTransformerConv(
                hidden_channels, hidden_channels // heads,
                heads=heads, concat=True, dropout=dropout))
        # Final layer
        self.convs.append(PyGTransformerConv(
            hidden_channels, out_channels,
            heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                if self.use_residual and i > 0:
                    x = x + x_prev
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 7.  GATv2  (Dynamic Attention)
# ══════════════════════════════════════════════════════════════════════════
class GATv2Encoder(nn.Module):
    """
    Variable-depth GATv2 with dynamic attention (Brody et al., 2022).

    Residual connections for num_layers > 2.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, heads: int = 4,
                 dropout: float = 0.5, num_layers: int = 2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = num_layers > 2

        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels,
                                    heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels,
                                        heads=heads, concat=True, dropout=dropout))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels,
                                    heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                if self.use_residual and i > 0:
                    x = x + x_prev
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ══════════════════════════════════════════════════════════════════════════
# 8.  LightGCN  (Recommendation-Tailored)
# ══════════════════════════════════════════════════════════════════════════
class LightGCNEncoder(nn.Module):
    """
    LightGCN encoder (He et al., 2020) — purpose-built for collaborative
    filtering / recommendation.

    Key differences from standard GCNs:
      • NO learnable weight matrices in the convolution layers.
      • NO non-linear activation functions.
      • The final embedding is the **mean** of embeddings across all K
        layers (including the initial embedding), which acts as a form
        of self-ensembling and avoids over-smoothing.

    Note: LightGCN intentionally omits dropout, weight decay, and
    residual connections — its layer-mean design already provides
    implicit regularization and skip connections.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.0, num_nodes: int = None):
        super().__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels

        # Learnable initial embedding (replaces feature transforms)
        if num_nodes is not None:
            self.embedding = nn.Embedding(num_nodes, out_channels)
            nn.init.xavier_uniform_(self.embedding.weight)
            self._use_embedding_table = True
        else:
            self.projection = nn.Linear(in_channels, out_channels, bias=False)
            self._use_embedding_table = False

        # LightGCN uses bare GCNConv layers with NO learnable weights.
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(out_channels, out_channels, add_self_loops=False)
            )

        # Freeze all convolution weights — LightGCN has no trainable
        # parameters inside the message-passing layers.
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    def forward(self, x, edge_index):
        # Initial embedding
        if self._use_embedding_table:
            z = self.embedding.weight  # (N, out_channels)
        else:
            z = self.projection(x)    # (N, out_channels)

        all_z = [z]  # collect embeddings from every layer

        for conv in self.convs:
            z = conv(z, edge_index)
            # No activation, no dropout — by design
            all_z.append(z)

        # Final embedding = mean of all layer outputs (self-ensemble)
        # This IS the implicit skip/residual mechanism in LightGCN.
        z_final = torch.stack(all_z, dim=0).mean(dim=0)
        return z_final


# ══════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY  —  single lookup table used by benchmark_all.py
# ══════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "GCN":            GCNEncoder,
    "GNN":            GraphConvEncoder,
    "GAT":            GATEncoder,
    "GraphSAGE":      GraphSAGEEncoder,
    "GIN":            GINEncoder,
    "TransformerConv": TransformerEncoder,
    "GATv2":          GATv2Encoder,
    "LightGCN":       LightGCNEncoder,
}
