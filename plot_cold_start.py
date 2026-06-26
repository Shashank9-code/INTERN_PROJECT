# === DELIVERABLE B: plot_cold_start.py ===
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')

import seaborn as sns

os.makedirs('defense_outputs', exist_ok=True)

models   = ['GCN','GNN','GAT','GraphSAGE','GIN',
            'TransformerConv','GATv2','LightGCN']
low_deg  = [0.571, 0.688, 0.221, 0.488, 0.729,
            0.429, 0.283, 0.263]
high_deg = [0.369, 0.498, 0.345, 0.358, 0.513,
            0.344, 0.377, 0.305]

x = np.arange(len(models))
width = 0.35

palette = sns.color_palette('deep', 2)

fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

bars1 = ax.bar(x - width/2, low_deg,  width, label='Low-Degree (New Users)',
               color=palette[0], edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, high_deg, width, label='High-Degree (Established Users)',
               color=palette[1], edgecolor='white', linewidth=0.5)

# Data labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h,
            f'{h:.3f}', fontsize=8, ha='center', va='bottom')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h,
            f'{h:.3f}', fontsize=8, ha='center', va='bottom')

# Threshold line
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8,
           label='0.5 threshold', zorder=0)

# Annotate GIN Low-Deg bar (index 4)
gin_bar = bars1[4]
ax.annotate('\u2605 Best Cold-Start',
            xy=(gin_bar.get_x() + gin_bar.get_width()/2, gin_bar.get_height()),
            xytext=(0, 12), textcoords='offset points',
            color='red', fontsize=8, ha='center', va='bottom',
            fontweight='bold')

# Annotate GAT Low-Deg bar (index 2)
gat_bar = bars1[2]
ax.annotate('\u26a0 Attention Collapse',
            xy=(gat_bar.get_x() + gat_bar.get_width()/2, gat_bar.get_height()),
            xytext=(0, 12), textcoords='offset points',
            color='red', fontsize=8, ha='center', va='bottom',
            fontweight='bold')

ax.set_xlabel('Model Architecture', fontsize=12)
ax.set_ylabel('Recall@10', fontsize=12)
ax.set_title('Cold-Start Analysis: Low-Degree vs High-Degree Users', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 0.85)

plt.tight_layout()
plt.savefig('defense_outputs/cold_start_chart.png')
print('Saved: defense_outputs/cold_start_chart.png')
