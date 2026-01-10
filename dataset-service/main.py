from fastapi import FastAPI, Query
from pydantic import BaseModel
from data import dataset, get_embedding, index
from datetime import datetime
from typing import Optional
import numpy as np
import faiss

app = FastAPI(title="Dataset Service")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/data")
def filter_data(
    category: Optional[str] = None,
    status: Optional[str] = None, 
    priority: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    filtered = dataset
    if category: filtered = [r for r in filtered if r["category"] == category]
    if status: filtered = [r for r in filtered if r["status"] == status]
    if priority: filtered = [r for r in filtered if r["priority"] == priority]
    if start_date:
        sd = datetime.fromisoformat(start_date)
        filtered = [r for r in filtered if datetime.fromisoformat(r["created_at"]) >= sd]
    return filtered[:20]

class SemanticRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search/semantic")
def semantic_search(req: SemanticRequest):
    q_emb = np.array([get_embedding(req.query)], dtype=np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, req.top_k)
    return [
        {"record": dataset[i], "score": float(score)} 
        for i, score in zip(indices[0], scores[0])
    ]
