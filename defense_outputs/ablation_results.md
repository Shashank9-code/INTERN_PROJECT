# Feature Engineering Ablation Study

**Target Model:** GINEncoder  |  **Loss:** BPR  |  **Epochs:** 200  |  **Patience:** 25

| Run   | Features                    |      AUC |   nDCG@10 |
|:------|:----------------------------|---------:|----------:|
| Run 1 | Baseline: Identity Only     | 0.935174 |  0.419049 |
| Run 2 | + Local Topology: Degree    | 0.943909 |  0.421459 |
| Run 3 | + Global Topology: PageRank | 0.945743 |  0.430198 |
