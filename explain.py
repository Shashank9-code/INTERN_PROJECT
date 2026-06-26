import os
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.utils import k_hop_subgraph
from data_utils import load_facebook_dataset, prepare_splits
from models import MODEL_REGISTRY

def main():
    print("=" * 60)
    print("  EXPLAINABLE AI (XAI) MODULE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    data = load_facebook_dataset(root="./data")
    train_data, _, test_data = prepare_splits(data)
    train_data, test_data = train_data.to(device), test_data.to(device)

    # 2. Load Model
    model_name = "GATv2"
    model_path = f"defense_outputs/best_model_{model_name.lower().replace(' ', '_')}.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint {model_path} not found.")
        print("Please run `python benchmark_all.py` first to generate the models.")
        return

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(train_data.x.size(1), 128, 64, heads=4, dropout=0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Extract Attention Weights from final layer
    print(f"  🔍  Extracting attention weights from {model_name}...")
    with torch.no_grad():
        x = F.dropout(train_data.x, p=model.dropout, training=False)
        for i, conv in enumerate(model.convs[:-1]):
            x_prev = x
            x = conv(x, train_data.edge_index)
            x = F.elu(x)
            if model.use_residual and i > 0:
                x = x + x_prev
            x = F.dropout(x, p=model.dropout, training=False)
        
        # Last layer
        z, (edge_index, alpha) = model.convs[-1](x, train_data.edge_index, return_attention_weights=True)
    
    # Average attention across heads
    alpha = alpha.mean(dim=-1).cpu().numpy()
    edge_index = edge_index.cpu()
    z_cpu = z.cpu()

    # 4. Find a True Positive (Successful Recommendation)
    pos_test_edges = test_data.edge_label_index[:, test_data.edge_label == 1.0].cpu()
    scores = (z_cpu[pos_test_edges[0]] * z_cpu[pos_test_edges[1]]).sum(dim=-1)
    
    # Pick the top-scoring true positive edge
    top_indices = torch.argsort(scores, descending=True)
    best_idx = top_indices[0].item()
    target_u = pos_test_edges[0, best_idx].item()
    target_v = pos_test_edges[1, best_idx].item()
    
    print(f"  ✅  Target User (True Positive Recommendation): {target_u} -> {target_v}")
    print(f"      Model Confidence Score: {scores[best_idx]:.4f}")

    # 5. Extract 2-hop Subgraph
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        target_u, num_hops=2, edge_index=edge_index, relabel_nodes=True
    )
    sub_alpha = alpha[edge_mask.cpu()]
    
    # 6. Build NetworkX Graph
    G = nx.DiGraph()
    for i in range(len(subset)):
        G.add_node(i, original_id=subset[i].item())
        
    src = sub_edge_index[0].numpy()
    dst = sub_edge_index[1].numpy()
    weights = sub_alpha
    
    for s, d, w in zip(src, dst, weights):
        G.add_edge(s, d, weight=w)
        
    # 7. Visualize Subgraph
    print("  🎨  Generating visualization...")
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    node_colors = []
    for n in G.nodes():
        orig = G.nodes[n]['original_id']
        if orig == target_u:
            node_colors.append('#e74c3c') # Red for target
        elif orig == target_v:
            node_colors.append('#2ecc71') # Green for recommended
        else:
            node_colors.append('#3498db') # Blue for others
            
    # Normalize edge thickness based on attention
    w_vals = np.array([d['weight'] for u, v, d in G.edges(data=True)])
    w_vals = (w_vals - w_vals.min()) / (w_vals.max() - w_vals.min() + 1e-9)
    edge_widths = 1.0 + 5.0 * w_vals  # Scale from 1 to 6
    
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=edge_widths, alpha=0.7, arrowsize=10, connectionstyle="arc3,rad=0.1")
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Target User ({target_u})', markerfacecolor='#e74c3c', markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'Recommended Friend ({target_v})', markerfacecolor='#2ecc71', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Context Neighbors', markerfacecolor='#3498db', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    plt.title(f"XAI: GATv2 Attention Weights Subgraph (2-Hop)", fontsize=15, fontweight="bold")
    plt.axis("off")
    
    os.makedirs("defense_outputs", exist_ok=True)
    save_path = "defense_outputs/attention_subgraph.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  📊  Explainable AI subgraph saved → {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
