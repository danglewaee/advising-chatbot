import json, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROC = Path("data/processed/chunks.json")
chunks = json.loads(PROC.read_text(encoding="utf-8"))

print(f"Embedding {len(chunks)} chunks with local model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
# Compute embeddings for chunks
vecs = np.array([model.encode(c["text"]) for c in chunks], dtype="float32")
# normalize vectors to unit length so inner-product (cosine) search works
norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
vecs = vecs / norms

dim = vecs.shape[1]
# use inner-product index on normalized vectors (equivalent to cosine)
index = faiss.IndexFlatIP(dim)
index.add(vecs)

Path("data/indices").mkdir(parents=True, exist_ok=True)
faiss.write_index(index, "data/indices/faiss.index")
Path("data/indices/meta.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

print(f"✅ indexed {len(chunks)} chunks locally -> data/indices/faiss.index")

# Compute and persist full-key embeddings (one vector per unique orig_key)
# so retrieval can quickly match queries to canonical JSON keys without
# re-encoding them on every request.
key_names = []
for c in chunks:
	ok = c.get("orig_key")
	if ok and ok not in key_names:
		key_names.append(ok)

if key_names:
	print(f"Embedding {len(key_names)} unique keys...")
	kvecs = model.encode(key_names, convert_to_numpy=True)
	kvecs = np.array(kvecs, dtype="float32")
	knorms = np.linalg.norm(kvecs, axis=1, keepdims=True) + 1e-10
	kvecs = kvecs / knorms
	try:
		out_dir = Path("data/indices")
		out_dir.mkdir(parents=True, exist_ok=True)
		np.save(str(out_dir / "key_name_embs.npy"), kvecs)
		(out_dir / "key_name_meta.json").write_text(json.dumps({"keys": key_names}, ensure_ascii=False), encoding="utf-8")
		print(f"✅ wrote {len(key_names)} key-name embeddings -> data/indices/key_name_embs.npy")
	except Exception as e:
		print("Failed to persist key-name embeddings:", e)
