# GNN-Based Friend Recommendation System

A comprehensive Graph Neural Network implementation for friend recommendation via link prediction. This project explores multiple GNN architectures (GCN, GAT, GraphConv) with advanced techniques including contrastive learning, adversarial robustness, and cold-start scenarios.

## 🎯 Project Overview

This internship project implements state-of-the-art GNN architectures for social network friend recommendation:

- **GCN (Graph Convolutional Network)** - Symmetric-normalized spectral convolution
- **GAT (Graph Attention Network)** - Multi-head attention over neighbors
- **GraphConv (Vanilla MPNN)** - Basic message passing framework
- **Contrastive Learning & Adversarial Training** - Robust recommendations
- **Cold-start Analysis** - Performance with new users

## 📁 Project Structure

```
INTERN_PROJECT/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── standalone_tutorials/              # Phase 1: Individual implementations
│   ├── friend_recommendation_gcn.py
│   ├── friend_recommendation_gat.py
│   ├── friend_recommendation_gnn.py
│   └── ablation_study_layers.py
├── models.py                          # Modular GNN architectures (Phase 2)
├── losses.py                          # BPR Loss & custom loss functions
├── data_utils.py                      # Data loading and preprocessing
├── evaluate.py                        # Evaluation metrics (Recall, MRR, NDCG)
├── ablation_study.py                  # Layer depth analysis
├── benchmark_all.py                   # Comprehensive benchmarking
├── fastapi_inference.py               # REST API for inference
├── explain.py                         # Model explainability
├── plot_cold_start.py                 # Cold-start performance analysis
├── tsne_dashboard.py                  # t-SNE visualization
├── tsne_dashboard.png                 # Visualization output
├── data/                              # Dataset directory
├── report/                            # Analysis reports
├── defense_outputs/                   # Adversarial robustness results
└── tsne_individual/                   # Individual t-SNE plots
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Shashank9-code/INTERN_PROJECT.git
cd INTERN_PROJECT

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

**Phase 1 - Educational Standalone Tutorials:**
```bash
python standalone_tutorials/friend_recommendation_gcn.py
python standalone_tutorials/friend_recommendation_gat.py
python standalone_tutorials/friend_recommendation_gnn.py
```

**Phase 2 - Modular Pipeline:**
```bash
# Run comprehensive benchmark
python benchmark_all.py

# Run ablation study (layer depth analysis)
python ablation_study.py

# Evaluate models
python evaluate.py

# Cold-start scenario analysis
python plot_cold_start.py

# Generate visualizations
python tsne_dashboard.py
```

**Inference & Deployment:**
```bash
# Start FastAPI inference server
python fastapi_inference.py
```

## 📊 Key Features

### Architectures Implemented
| Architecture | File | Loss Function | Use Case |
|-------------|------|--------------|----------|
| GCN | `friend_recommendation_gcn.py` | BCE Loss (Classification) | Symmetric graph convolution |
| GAT | `friend_recommendation_gat.py` | BCE Loss | Attention-based node aggregation |
| GraphConv | `friend_recommendation_gnn.py` | BCE Loss | Basic MPNN framework |
| Modular GNN | `models.py` | BPR Loss (Ranking) | Production-grade recommendation |

### Advanced Features
- **Contrastive Learning** - Improved embedding quality
- **Adversarial Robustness** - Defense against adversarial attacks
- **Cold-start Analysis** - Performance with new users
- **Model Explainability** - Understand prediction reasoning
- **REST API** - Deploy as web service

## 📈 Evaluation Metrics

Models are evaluated using ranking-based metrics:
- **Recall@K** - Proportion of relevant items in top-K recommendations
- **MRR** - Mean Reciprocal Rank
- **NDCG** - Normalized Discounted Cumulative Gain
- **AUC** - Area Under ROC Curve

## 📚 Key Differences

### Phase 1 (Standalone Tutorials)
- Self-contained, monolithic implementations
- **Loss:** Binary Cross-Entropy (Classification-based)
- **Purpose:** Educational, easy to understand individual architectures

### Phase 2 (Modular Pipeline)
- Refactored, production-ready code
- **Loss:** Bayesian Personalized Ranking (Recommendation-based)
- **Purpose:** Enterprise deployment, benchmarking

## 🔧 Dependencies

- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural networks
- **DGL** - Alternative graph library
- **Scikit-learn** - Machine learning utilities
- **NumPy, Pandas** - Data processing
- **Matplotlib, Seaborn** - Visualization
- **FastAPI** - REST API framework

See `requirements.txt` for complete dependencies.

## 📖 Documentation

For detailed information:
- Check `standalone_tutorials/README.md` for Phase 1 architecture details
- Review individual Python files for implementation specifics
- See `report/` directory for analysis and findings

## 🎓 Academic Context

This project implements techniques from:
- Kipf & Welling (2017) - Semi-Supervised Classification with GCN
- Veličković et al. (2018) - Graph Attention Networks
- Rendle et al. (2009) - BPR: Bayesian Personalized Ranking

## 📝 Author

**Shashank Prabhakar**  
Internship Project - Graph Neural Networks for Recommendation Systems

## 📄 License

Open source - feel free to use for educational and research purposes.

## 🤝 Contributing

Suggestions and improvements are welcome! Open an issue or submit a pull request.

---

**Last Updated:** June 2026  
**Repository:** https://github.com/Shashank9-code/INTERN_PROJECT
