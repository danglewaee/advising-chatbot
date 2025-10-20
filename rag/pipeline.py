import re
import faiss, json, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

INDEX_PATH = Path("data/indices/faiss.index")
META_PATH  = Path("data/indices/meta.json")

if not INDEX_PATH.exists() or not META_PATH.exists():
    raise SystemExit("❌ Missing index/meta. Run embed_index.py first.")

index = faiss.read_index(str(INDEX_PATH))
meta  = json.loads(META_PATH.read_text(encoding="utf-8"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Helpers
COURSE_CODE = r"\\b(COMPSCI|INFO|MATH|STAT|PHYS|CS)\\s?-?\\d{2,3}[A-Z]?\\b"
PREREQ_KEYS = re.compile(r"prereq|pre-?req|eligibil|requirement|must have|prior to|recommended", re.I)

def split_sentences(text: str):
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
    clean = []
    for s in parts:
        s = re.sub(r"[^A-Za-z0-9.,;:/()\\-\\s]", "", s)
        if 20 < len(s) < 500:
            clean.append(s.strip())
    return clean


def score_sentence(q_emb, sent: str):
    s_emb = embedder.encode(sent)
    sim = float(np.dot(q_emb, s_emb) / (np.linalg.norm(q_emb)*np.linalg.norm(s_emb) + 1e-8))
    has_key = 1.0 if PREREQ_KEYS.search(sent) else 0.0
    n_codes = len(re.findall(COURSE_CODE, sent))
    # parameters: weights
    return 0.6 * sim + 0.3 * has_key + 0.1 * min(n_codes, 3)

def retrieve(query: str, k=6):
    qv = np.array([embedder.encode(query)], dtype="float32")
    D, I = index.search(qv, k)
    return [meta[i] for i in I[0]]

def extractive_answer(query: str):
    hits = retrieve(query, k=8)
    q_emb = embedder.encode(query)

    # collecting chunks
    candidates = []
    for h in hits:
        for s in split_sentences(h["text"]):
            if len(s) > 20:  
                candidates.append((s, h["source"]))

    if not candidates:
        return "Sorry, I couldn't find information in the collected documents.", list(set(h["source"] for h in hits))

    # scoring & picking top
    scored = [(score_sentence(q_emb, s), s, src) for (s, src) in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:5]

    # filtering
    filtered = []
    seen = set()
    for _, s, src in top:
        key = (s[:80], src)
        if key in seen: 
            continue
        seen.add(key)
        filtered.append((s, src))

    # prompt answering
    lines = []
    used_sources = set()
    for s, src in filtered:
        lines.append(f"- {s}")
        used_sources.add(src)
    if not lines:
        # fallback:first top sentence
        lines = ["(No explicit prerequisite sentence found; showing the most relevant context)", "-", hits[0]["text"][:300] + "..."]
        used_sources = {hits[0]["source"]}

    answer = "Here’s what I found about prerequisites:\n" + "\\n".join(lines)
    return answer, sorted(used_sources)

def answer(query: str):
    ans, srcs = extractive_answer(query)
    return f"{ans}\\n\\n(Sources: {', '.join(srcs)})"

if __name__ == "__main__":
    q = input("Ask a question: ")
    print(answer(q))
