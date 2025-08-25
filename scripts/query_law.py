import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Paths
INDEX_DIR = "law_index_faiss"
INDEX_PATH = f"{INDEX_DIR}/faiss_index.bin"
META_PATH = f"{INDEX_DIR}/metadata.json"

# Top-k retrieval
TOP_K = 10
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load FAISS index and metadata
def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# Embed query
def embed_query(query, model):
    return np.array([model.encode(query, normalize_embeddings=True)], dtype="float32")

# Retrieve top-k relevant documents
def retrieve(query, index, metadata, model, k=TOP_K):
    q_emb = embed_query(query, model)
    D, I = index.search(q_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1: 
            continue
        results.append({
            "score": float(dist),
            "text": metadata[str(idx)]["text"][:1200]  # preview for context
        })
    return results

# Ask Gemma 2B with refined prompt
def ask_gemma(query, context):
    prompt = f"""
You are an expert legal assistant. Follow these rules:

1. Provide the answer **directly** in the first line (no filler words).
2. Provide **detailed reasoning** strictly based on the provided context.
3. Cite the source of each point clearly (law name, section, chapter, or JSON file).
4. If multiple relevant sections exist, format reasoning as bullet points.
5. If context is empty or irrelevant, respond: "No relevant information found."

Context:
{context}

Question: {query}
Answer:
"""
    resp = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()

# Main loop
if __name__ == "__main__":
    print("[INFO] Legal semi-RAG system ready!")
    model = SentenceTransformer(EMBED_MODEL)
    index, metadata = load_index()

    while True:
        query = input("\nEnter your legal query (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = retrieve(query, index, metadata, model)
        if not results:
            print("\n⚖️ No relevant sections found.\n")
            continue

        # Debug: show top-k retrieval
        print("\n[DEBUG] Retrieved sections:")
        for r in results:
            print(f"- Score: {r['score']:.4f}, Preview: {r['text'][:150]}...")

        context = "\n\n".join([r["text"] for r in results])
        answer = ask_gemma(query, context)

        print("\n⚖️ Answer:\n", answer, "\n")
