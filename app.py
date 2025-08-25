import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import ollama

# Paths
INDEX_DIR = "law_index_faiss"
INDEX_PATH = f"{INDEX_DIR}/faiss_index.bin"
META_PATH = f"{INDEX_DIR}/metadata.json"
TOP_K = 10
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load index & metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model
model = SentenceTransformer(EMBED_MODEL)

# Functions
def embed_query(query):
    return np.array([model.encode(query, normalize_embeddings=True)], dtype="float32")

def retrieve(query):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, TOP_K)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append(metadata[str(idx)]["text"][:1200])
    return results

def ask_gemma(query, context):
    prompt = f"""
You are an expert legal assistant. Answer concisely first, then reason based ONLY on the context.
Cite the source of each point (law name, section, chapter, or JSON file). If context is irrelevant, say: "No relevant information found."

Context:
{context}

Question: {query}
Answer:
"""
    resp = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()

# Streamlit UI
st.title("⚖️ Legal QA Assistant (RAG-based)")
query = st.text_input("Enter your legal query:")

if st.button("Get Answer") and query:
    with st.spinner("Retrieving relevant laws..."):
        retrieved = retrieve(query)
        context = "\n\n".join(retrieved)
    st.subheader("Retrieved Sections")
    for r in retrieved:
        st.write("- ", r[:200], "...")
    
    with st.spinner("Generating answer..."):
        answer = ask_gemma(query, context)
    st.subheader("Answer")
    st.write(answer)
