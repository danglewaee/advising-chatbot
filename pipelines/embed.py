from __future__ import annotations
import os, pickle
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[1]  
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "meta.pkl"
CONF_PATH = OUT_DIR / "config.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

def _read_files() -> List[Dict[str, Any]]:
    docs = []
    for ext in ("*.txt", "*.md"):
        for p in DATA_DIR.rglob(ext):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if txt.strip():
                    docs.append({"path": str(p), "text": txt})
            except Exception:
                pass
    return docs

def _chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        window = tokens[i:i+max_tokens]
        chunks.append(enc.decode(window))
        i += max_tokens - overlap
    return chunks

def _embed_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return vecs.astype("float32")

def build() -> None:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    docs = _read_files()
    if not docs:
        print(f"[embed] No docs in {DATA_DIR}. Add .txt/.md first.")
        return

    texts, metas = [], []
    for doc in docs:
        chunks = _chunk_text(doc["text"])
        for j, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"path": doc["path"], "chunk_id": j})

    X = _embed_texts(texts, model)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)
    with open(CONF_PATH, "wb") as f:
        pickle.dump({"embed_model_name": EMBED_MODEL_NAME, "dim": dim}, f)

    print(f"[embed] Indexed {len(texts)} chunks from {len}(docs) files.")
    print(f"[embed] Saved index -> {INDEX_PATH.name}, meta -> {META_PATH.name}, config -> {CONF_PATH.name}")

if __name__ == "__main__":
    build()
