# Semantic Search AI Agent System

## Architecture
dataset-service/ ← APIs + Embeddings + FAISS
agent-app/ ← NL → Filters → Smart Responses

## Setup
```bash
# Generate data
cd dataset-service
python generate_data.py
python data.py  # Create embeddings

# Terminal 1: Dataset Service
cd dataset-service
uvicorn main:app --port 8000

# Terminal 2: AI Agent
cd agent-app
uvicorn main:app --port 8001 --env-file .env