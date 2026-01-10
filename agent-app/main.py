from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. CREATE APP FIRST (CRITICAL!)
app = FastAPI(title="Semantic Search API")

# 2. Load data and model (put here, NOT imported)
model = SentenceTransformer('all-MiniLM-L6-v2')
dataset = json.load(open("data_with_embeddings.json"))
embeddings_np = np.array([r["embedding"] for r in dataset], dtype=np.float32)
faiss.normalize_L2(embeddings_np)
index = faiss.IndexFlatIP(384)  # MiniLM dim=384
index.add(embeddings_np)

def get_embedding(text):
    return model.encode([text])[0]

# 3. NOW define endpoints (AFTER app created)
@app.get("/health")
def health():
    return {"status": "healthy", "records": len(dataset)}

@app.get("/data")
def filter_data(
    category: Optional[str] = None,
    status: Optional[str] = None, 
    priority: Optional[str] = None
):
    filtered = dataset
    if category: 
        filtered = [r for r in filtered if r["category"] == category]
    if status: 
        filtered = [r for r in filtered if r["status"] == status]
    if priority: 
        filtered = [r for r in filtered if r["priority"] == priority]
    return filtered[:20]

class SemanticRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search/semantic")
def semantic_search(req: SemanticRequest):
    try:
        q_emb = np.array([get_embedding(req.query)], dtype=np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, min(req.top_k, len(dataset)))
        return [
            {"record": dataset[i], "score": float(score)} 
            for i, score in zip(indices[0], scores[0]) if i < len(dataset)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
