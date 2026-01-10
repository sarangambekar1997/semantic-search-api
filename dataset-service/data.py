import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()

model = SentenceTransformer(os.getenv("HF_MODEL", "all-MiniLM-L6-v2"))
dataset = json.load(open("data.json"))

# Generate embeddings (run once)
print("Generating embeddings...")
texts = [f"{r['title']}. {r['description']}. Category: {r['category']}" for r in dataset]
embeddings = model.encode(texts)
for i, emb in enumerate(embeddings):
    dataset[i]["embedding"] = emb.tolist()

# Save with embeddings
with open("data_with_embeddings.json", "w") as f:
    json.dump(dataset, f)

# FAISS Index
dataset = json.load(open("data_with_embeddings.json"))
embeddings_np = np.array([r["embedding"] for r in dataset], dtype=np.float32)
faiss.normalize_L2(embeddings_np)
index = faiss.IndexFlatIP(384)  # MiniLM dim=384
index.add(embeddings_np)

def get_embedding(text):
    return model.encode([text])[0]

print("Embeddings ready. Index built.")
