from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re

# Load environment
load_dotenv()

app = FastAPI(title="AI Agent App", description="Natural language → Dataset Service → Smart Response")

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8000")

def parse_query(query: str) -> Dict[str, str]:
    """
    Converts natural language to filter parameters
    "high priority open payment issues" → {"category": "Payment", "priority": "High", "status": "Open"}
    """
    filters = {}
    query_lower = query.lower()
    
    # Category detection
    category_map = {
        "payment": "Payment", "billing": "Payment", "charge": "Payment",
        "login": "Login", "account": "Login", "sign": "Login",
        "order": "Order", "purchase": "Order",
        "shipping": "Shipping", "delivery": "Shipping"
    }
    for keyword, category in category_map.items():
        if keyword in query_lower:
            filters["category"] = category
            break
    
    # Priority detection
    priority_map = {
        "high": "High", "urgent": "High", "critical": "High",
        "medium": "Medium", "normal": "Medium",
        "low": "Low"
    }
    for keyword, priority in priority_map.items():
        if keyword in query_lower:
            filters["priority"] = priority
            break
    
    # Status detection
    status_map = {
        "open": "Open", "pending": "Open",
        "closed": "Closed", "resolved": "Resolved"
    }
    for keyword, status in status_map.items():
        if keyword in query_lower:
            filters["status"] = status
            break
    
    # Date range detection
    days_match = re.search(r'last (\d+)', query_lower)
    if days_match:
        days = int(days_match.group(1))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        filters["start_date"] = start_date.isoformat()
        filters["end_date"] = end_date.isoformat()
    
    return filters

def generate_smart_response(query: str, data: list) -> str:
    """
    Generates human-readable summary with insights (no external LLM)
    """
    if not data:
        return "No matching tickets found for your query."
    
    count = len(data)
    categories = [r.get("category", "Unknown") for r in data]
    priorities = [r.get("priority", "Unknown") for r in data]
    statuses = [r.get("status", "Unknown") for r in data]
    
    # Key insights
    insights = []
    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    high_count = priorities.count("High")
    open_count = statuses.count("Open")
    
    if high_count > 0:
        insights.append(f"{high_count} high-priority")
    if open_count > 0:
        insights.append(f"{open_count} open")
    
    top_category = max(cat_counts.items(), key=lambda x: x[1])
    if top_category[1] > 1:
        insights.append(f"mostly {top_category[0].lower()}")
    
    # Build response
    response = f"Found {count} matching ticket{'s' if count > 1 else ''}."
    
    if insights:
        response += f" {', '.join(insights)}."
    
    # Recent trends
    recent_days = []
    for ticket in data[:5]:
        created = datetime.fromisoformat(ticket["created_at"])
        days_ago = (datetime.now() - created).days
        if days_ago <= 2:
            recent_days.append(ticket["title"][:30])
    
    if recent_days:
        response += f" Recent: {', '.join(recent_days)}."
    
    return response

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy", 
        "service_url": SERVICE_URL,
        "description": "AI Agent ready - natural language queries supported"
    }

@app.get("/agent")
async def agent_query(query: str):
    """
    Main endpoint: Natural language → Dataset Service → Smart response
    
    Example: /agent?query=high priority open payment issues from last 2 days
    """
    try:
        print(f"Processing query: {query}")
        
        # Step 1: Parse natural language → structured filters
        filters = parse_query(query)
        print(f"Parsed filters: {filters}")
        
        # Step 2: Call Dataset Service API
        if filters:
            # Filter-based search
            response = requests.get(
                f"{SERVICE_URL}/data", 
                params=filters, 
                timeout=10
            )
            data = response.json()
            strategy = "filter-based"
        else:
            # Semantic search fallback
            response = requests.post(
                f"{SERVICE_URL}/search/semantic",
                json={"query": query, "top_k": 5},
                timeout=10
            )
            data = response.json()
            strategy = "semantic"
        
        print(f"Found {len(data)} results using {strategy}")
        
        # Step 3: Generate intelligent response
        answer = generate_smart_response(query, data)
        
        return {
            "original_query": query,
            "search_strategy": strategy,
            "filters_used": filters,
            "result_count": len(data),
            "human_response": answer,
            "sample_results": data[:3]  # First 3 for preview
        }
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Dataset service unavailable: {str(e)}"}
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
