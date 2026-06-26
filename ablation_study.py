"""
# Feature Engineering Ablation Study
# ====================================================================
#
# Prove the necessity of the topological features (Degree and PageRank).
#
# RUNS:
# 1. Baseline: Identity matrix only
# 2. + Local Topology: Identity + Normalized Node Degree
# 3. + Global Topology: Identity + Node Degree + PageRank
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import set_seed, load_facebook_dataset, prepare_splits
from benchmark_all import train_model, TOP_K
from models import MODEL_REGISTRY
from evaluate import evaluate_all

def main():
    print("=" * 60)
    print("  FEATURE ENGINEERING ABLATION STUDY")
    print("=" * 60)
    
    # Setup
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    original_data = load_facebook_dataset(root="./data")
    num_nodes = original_data.num_nodes
    
    # In data_utils, the features are concatenated as:
    # [identity(num_nodes), degree(1), pagerank(1)]
    # We slice data.x to control which features are passed to the model.
    configs = [
        ("Run 1 (Baseline: Identity Only)", num_nodes),
        ("Run 2 (+ Local Topology: Degree)", num_nodes + 1),
        ("Run 3 (+ Global Topology: PageRank)", original_data.x.size(1))
    ]
    
    results = []
    
    for run_name, feature_dim in configs:
        print(f"\n🚀 {run_name}")
        print("━" * 40)
        
        # Clone data and slice features
        data = original_data.clone()
        data.x = data.x[:, :feature_dim]
        
        # Split data (deterministic due to set_seed inside prepare_splits)
        train_data, val_data, test_data = prepare_splits(data, seed=SEED)
        train_data, val_data, test_data = train_data.to(DEVICE), val_data.to(DEVICE), test_data.to(DEVICE)
        
        # Target Model: GINEncoder
        set_seed(SEED)
        model_cls = MODEL_REGISTRY["GIN"]
        model = model_cls(in_channels=feature_dim, hidden_channels=128, out_channels=64, dropout=0.5).to(DEVICE)
        
        # Train Model (uses exact same BPR loss, max 200 epochs, early stopping from benchmark_all)
        model, history = train_model("GIN", model, train_data, val_data)
        
        # Evaluate
        metrics = evaluate_all(model, test_data, train_data.edge_index, K=TOP_K)
        
        results.append({
            "Run": run_name.split(" (")[0], # just "Run 1", "Run 2"
            "Features": run_name.split("(")[1].replace(")", ""),
            "AUC": metrics["AUC"],
            f"nDCG@{TOP_K}": metrics[f"nDCG@{TOP_K}"]
        })
        
        print(f"  ✅ Test AUC: {metrics['AUC']:.4f}  |  nDCG@{TOP_K}: {metrics[f'nDCG@{TOP_K}']:.4f}")
        
    print("\n" + "=" * 60)
    print("  VISUALIZATION & EXPORT")
    print("=" * 60)
    
    os.makedirs("defense_outputs", exist_ok=True)
    df = pd.DataFrame(results)
    
    # 1. Export Table
    md_path = "defense_outputs/ablation_results.md"
    with open(md_path, "w") as f:
        f.write("# Feature Engineering Ablation Study\n\n")
        f.write("**Target Model:** GINEncoder  |  **Loss:** BPR  |  **Epochs:** 200  |  **Patience:** 25\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"  💾  Results table saved → {md_path}")
    
    # 2. Export Grouped Bar Chart
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, df["AUC"], width, label='Test AUC', color='#3498db', edgecolor='white')
    rects2 = ax.bar(x + width/2, df[f"nDCG@{TOP_K}"], width, label=f'Test nDCG@{TOP_K}', color='#2ecc71', edgecolor='white')
    
    ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Ablation Study (GINEncoder)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    
    # Use incremental labels for the x-axis
    x_labels = [
        "Baseline\n(Identity Matrix)", 
        "+ Local Topology\n(Normalized Degree)", 
        "+ Global Topology\n(PageRank)"
    ]
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1.1)
    
    # Add exact values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    autolabel(rects1)
    autolabel(rects2)
    
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    chart_path = "defense_outputs/ablation_chart.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  📊  Bar chart saved → {chart_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
