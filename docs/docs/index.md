# Semantic Search API 

FastAPI-powered semantic search engine for payment documents using vector embeddings.

# Features
- Natural language search ("urgent payment problem")
- Vector similarity ranking with scores
- Payment category document matching
- Top-K results (configurable)
<!-- - Docker-ready deployment -->

# Quick Start

```bash
# Clone & run
git clone https://github.com/sarangambekar1997/semantic-search-api.git
cd semantic-search
# docker-compose up --build

# Test search
curl -X POST "http://localhost:8001/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "urgent payment problem", "top_k": 5}