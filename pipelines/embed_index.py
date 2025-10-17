import json, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROC = Path("data/processed/chunks.json")
chunks = json.loads(PROC.read_text(encoding="utf-8"))

print(f"Embedding {len(chunks)} chunks with local model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = [model.encode(c["text"]) for c in chunks]

dim = len(vecs[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(vecs, dtype="float32"))

Path("data/indices").mkdir(parents=True, exist_ok=True)
faiss.write_index(index, "data/indices/faiss.index")
Path("data/indices/meta.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

print(f"✅ indexed {len(chunks)} chunks locally -> data/indices/faiss.index")
