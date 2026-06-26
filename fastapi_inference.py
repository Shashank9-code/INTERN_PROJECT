"""
# GNN Link Prediction — Production Deployment API
# ====================================================================
#
# This script hosts the trained GINEncoder model using FastAPI, 
# exposing a production-ready REST endpoint for real-time friend 
# recommendations.
#
# HOW TO RUN THE SERVER LOCALLY:
# --------------------------------------------------------------------
# 1. Install the required web server dependencies (if not installed):
#    $ pip install fastapi uvicorn
#
# 2. Start the FastAPI server using Uvicorn:
#    $ uvicorn fastapi_inference:app --reload
#
# 3. Test the API in your browser or via curl:
#    Open your browser and navigate to: 
#    http://127.0.0.1:8000/recommend/203
#
#    Or test via Terminal:
#    $ curl http://127.0.0.1:8000/recommend/203
"""

import time
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from data_utils import load_facebook_dataset, prepare_splits
from models import MODEL_REGISTRY

# ══════════════════════════════════════════════════════════════════════════
#  1. GLOBAL INITIALIZATION (Executed once at startup)
# ══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Friend Recommendation Engine", 
              description="Real-time GNN Link Prediction using FastAPI")

print("Loading SNAP Facebook Dataset...")
# We use the CPU for inference as GNNs on small graphs are extremely fast
device = torch.device("cpu") 

data = load_facebook_dataset(root="./data")
# To mirror the training conditions perfectly, we extract the training edges 
# to mask out the users' existing friends during recommendation.
train_data, val_data, test_data = prepare_splits(data)

# Pre-build adjacency list for O(1) friend lookups
train_adj = {}
src_train = train_data.edge_index[0].numpy()
dst_train = train_data.edge_index[1].numpy()
for s, d in zip(src_train, dst_train):
    train_adj.setdefault(s, set()).add(d)

print("Loading pre-trained GINEncoder...")
# From benchmark_results.csv, GIN was the top performer (AUC: ~0.941).
model_name = "GIN"
model_path = "defense_outputs/best_model_gin.pt"

model_cls = MODEL_REGISTRY[model_name]
model = model_cls(in_channels=train_data.x.size(1), 
                  hidden_channels=128, 
                  out_channels=64, 
                  dropout=0.5).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ Model loaded successfully and ready for inference!")
except Exception as e:
    print(f"❌ Failed to load model weights: {e}")
    print("Please ensure you have run benchmark_all.py to generate defense_outputs/best_model_gin.pt")


# ══════════════════════════════════════════════════════════════════════════
#  2. API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
@app.get("/")
def health_check():
    return {"status": "online", "model": model_name, "nodes": data.num_nodes}

@app.get("/recommend/{user_id}")
def recommend_friends(user_id: int):
    """
    Generate Top-10 friend recommendations for a given user ID.
    """
    start_time = time.time()
    
    if user_id < 0 or user_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in graph.")

    # 1. Generate node embeddings for the entire graph
    with torch.no_grad():
        z = model(train_data.x, train_data.edge_index)
    
    # 2. Extract target user's embedding and calculate dot-product similarity
    user_emb = z[user_id]
    scores = (z * user_emb).sum(dim=-1) # (N,) array of similarity scores
    
    # 3. Mask out the user's existing friends and themselves
    existing_friends = train_adj.get(user_id, set()) | {user_id}
    mask_indices = list(existing_friends)
    scores[mask_indices] = float("-inf")
    
    # 4. Get the Top 10 recommendations
    top_k = 10
    top_scores, top_indices = torch.topk(scores, k=min(top_k, data.num_nodes))
    
    # Format the results
    recommendations = []
    for rank, (rec_id, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
        recommendations.append({
            "rank": rank + 1,
            "friend_id": rec_id,
            "similarity_score": round(score, 4)
        })
        
    execution_time_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "user_id": user_id,
        "existing_friends_count": len(existing_friends) - 1, # minus self
        "recommendations": recommendations,
        "inference_time_ms": execution_time_ms,
        "model_used": model_name
    }
