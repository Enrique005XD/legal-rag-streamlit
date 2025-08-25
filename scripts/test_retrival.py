import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "law_index_faiss/faiss_index.bin"
META_PATH = "law_index_faiss/metadata.json"

# Load embedding model
print("[INFO] Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
print("[INFO] Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def search(query, k=5):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        # FIX: handle both list-style and dict-style metadata
        key = str(idx) if isinstance(metadata, dict) else idx
        section = metadata[key]

        results.append({
            "rank": i + 1,
            "score": float(distances[0][i]),
            "source": section.get("source", "unknown"),
            "text": section.get("text", "")[:300] + "..."
        })
    return results



if __name__ == "__main__":
    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        hits = search(q, k=5)
        if not hits:
            print("⚖️ No matches found.")
        else:
            print(f"\n⚖️ Top {len(hits)} matches:")
            for h in hits:
                print(f"[{h['rank']}] (score={h['score']:.4f}) {h['source']}")
                print(f"   {h['text']}\n")
