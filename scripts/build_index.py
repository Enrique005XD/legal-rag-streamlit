import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Folders
DATA_DIR = "data/"  # 8 section-based JSONs
QA_DIR = "qa/"      # 3 Q/A JSONs
INDEX_DIR = "law_index_faiss"
os.makedirs(INDEX_DIR, exist_ok=True)

# Paths
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")

# Embedding model
EMBED_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

docs, metadata = [], {}
doc_id = 0

# Normalize section-based JSON entries
def normalize_section_entry(law, entry):
    if law in ["cpc.json", "ida.json", "mva.json"]:
        section = entry.get("section", "")
        title = entry.get("title", "")
        desc = entry.get("description", "")
        return f"[{law}] Section {section}: {title}\n{desc}"

    elif law in ["crpc.json", "hma.json", "iea.json", "nia.json"]:
        chap = entry.get("chapter", "")
        section = entry.get("section", "")
        title = entry.get("section_title", "")
        desc = entry.get("section_desc", "")
        return f"[{law}] Chapter {chap}, Section {section}: {title}\n{desc}"

    elif law == "ipc.json":
        chap = entry.get("chapter", "")
        chap_title = entry.get("chapter_title", "")
        title = entry.get("section_title", "")
        desc = entry.get("section_desc", "")
        return f"[{law}] Chapter {chap}: {chap_title}\nSection: {title}\n{desc}"

    return None

# Normalize Q/A JSON entries
def normalize_qa_entry(law, entry):
    question = entry.get("question", "")
    answer = entry.get("answer", "")
    if question and answer:
        return f"[{law}] Q: {question}\nA: {answer}"
    return None

# Load and index JSONs from a folder
def load_json_folder(folder, is_qa=False):
    global doc_id
    for file in os.listdir(folder):
        if file.endswith(".json"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            law_name = file.replace(".json", "")
            if isinstance(data, list):
                for entry in data:
                    text = normalize_qa_entry(law_name, entry) if is_qa else normalize_section_entry(file, entry)
                    if text and len(text.strip()) > 20:
                        docs.append(text)
                        metadata[doc_id] = {"file": file, "text": text[:500]}  # store preview
                        doc_id += 1
            else:
                print(f"[WARN] {file} not list-based, skipping")

# Load section-based JSONs
load_json_folder(DATA_DIR, is_qa=False)
# Load Q/A JSONs
load_json_folder(QA_DIR, is_qa=True)

print(f"[INFO] Total docs collected: {len(docs)}")
if not docs:
    raise ValueError("❌ No text extracted from JSON files!")

# Create embeddings
embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("[SUCCESS] FAISS index and metadata saved for all JSONs.")
