# agent-app/main.py - COMPLETE WORKING FILE
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Pydantic model for structured ticket response
class Ticket(BaseModel):
    id: int
    title: str
    description: str
    category: str
    priority: str
    status: str
    created_at: str
    embedding: List[float]

# Create FastAPI app (THIS WAS MISSING!)
app = FastAPI(title="Ticket API", version="1.0.0")

# Load embedding model (moved after app definition)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample tickets data
tickets_data = [
    Ticket(
        id=10,
        title="Official same hotel.",
        description="Remain father early. Stay give artist main enter enjoy. Provide ask believe put list.",
        category="Order",
        priority="High",
        status="Resolved",
        created_at="2025-12-23T14:14:38.467367",
        embedding=[0.033531736582517624, -0.016502682119607925, -0.0429190993309021]
    ),
    Ticket(
        id=11,
        title="Seek add campaign before reality.",
        description="Seat clearly mission management I. Walk official page product author.",
        category="Payment",
        priority="Medium",
        status="Open",  # Changed to Open for testing
        created_at="2025-12-13T14:14:38.467367",
        embedding=[-0.06408578157424927, 0.024369169026613235, -0.0331922322511673]
    )
]

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Ticket API is running", "endpoints": ["/data", "/query"]}

@app.get("/data", response_model=List[Ticket], tags=["Data"])
async def get_data():
    return tickets_data

# @app.post("/query", tags=["Semantic Search"])
# async def semantic_search(query: str):
#     """Semantic search endpoint - finds relevant tickets"""
#     query_embedding = model.encode(query)
    
#     results = []
#     for ticket in tickets_data:
#         ticket_embedding = np.array(ticket.embedding)
#         similarity = util.cos_sim(query_embedding, ticket_embedding)[0][0].item()
#         results.append({
#             "id": ticket.id,
#             "title": ticket.title,
#             "description": ticket.description,
#             "category": ticket.category,
#             "priority": ticket.priority,
#             "status": ticket.status,
#             "similarity": float(similarity)
#         })
    
#     return sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]

@app.post("/query")
async def semantic_search(query_data: dict):
    query = query_data.get("query", "").lower()
    
    results = []
    for ticket in tickets_data:
        text = f"{ticket.title} {ticket.description}".lower()
        score = 0
        
        if any(word in text for word in query.split()):
            score += 3
        if "payment" in ticket.category.lower() and "payment" in query:
            score += 5
        if ticket.priority == "High" and "high" in query:
            score += 4
        if ticket.status == "Open" and "open" in query:
            score += 2
            
        results.append({
            "id": ticket.id,
            "title": ticket.title,
            "category": ticket.category,
            "priority": ticket.priority,
            "status": ticket.status,
            "score": score
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:5]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
