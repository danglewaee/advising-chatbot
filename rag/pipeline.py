import os

# Prevent HuggingFace `tokenizers` from using parallelism in child processes
# when the interpreter forks. This must be set before importing libraries that
# instantiate tokenizers or create threads (sentence-transformers, tokenizers,
# transformers, etc.). Disabling or explicitly setting these prevents the
# "current process just got forked" warning and avoids deadlocks.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Optionally limit OpenMP/MKL threads to reduce thread contention
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# to import load_json from pipelines/json_loader.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import faiss, json, numpy as np
from sentence_transformers import SentenceTransformer
from pipelines.json_loader import load_json
from pathlib import Path
import dotenv
from openai import OpenAI
# Initialize OpenAI client only if an API key is available (from .env or env)
_OPENAI_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = None
if _OPENAI_KEY:
    try:
        client = OpenAI(api_key=_OPENAI_KEY)
    except Exception:
        client = None

INDEX_PATH = Path("data/indices/faiss.index")
META_PATH  = Path("data/indices/meta.json")

if not INDEX_PATH.exists() or not META_PATH.exists():
    raise SystemExit("❌ Missing index/meta. Run embed_index.py first.")

index = faiss.read_index(str(INDEX_PATH))
meta  = json.loads(META_PATH.read_text(encoding="utf-8"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Helpers
# Course-code regex: use word boundaries so lexical matches work as expected
COURSE_CODE = r"\b(CICS|COMPSCI|INFO|MATH|STAT|PHYS|CS)\s?-?\d{2,3}[A-Z]?\b"
PREREQ_KEYS = re.compile(r"prereq|pre-?req|eligibil|requirement|must have|prior to|recommended", re.I)


def generate_course_variants(code: str):
    """Given a matched course code like 'CICS 160' or 'CS160', return a list
    of lowercase variants to search for in text (e.g. ['cics 160','compsci 160','cs 160','cs160']).
    """
    if not code:
        return []
    # extract numeric part
    m = re.search(r"\d{2,3}[A-Z]?", code)
    if not m:
        return [code.lower()]
    num = m.group(0)
    variants = [f"cics {num}", f"compsci {num}", f"cs {num}", f"compsci{num}", f"cs{num}", f"cics{num}"]
    return [v.lower() for v in variants]


def load_all_documents(data_dir: str = "data/processed/json"):
    """Load and return a flat list of Document objects from JSON files.

    Uses `pipelines.json_loader.load_json` which returns a list of langchain
    Document-like objects. Ensures each Document has a metadata dict with a
    'source' fallback.
    """
    docs = []
    p = Path(data_dir)
    for f in p.glob("*.json"):
        try:
            file_docs = load_json(str(f))
        except Exception:
            continue
        for d in file_docs:
            # ensure metadata exists and include filename as fallback source
            if not isinstance(getattr(d, "metadata", None), dict):
                d.metadata = {"source": f.name}
            if not d.metadata.get("source"):
                d.metadata["source"] = f.name
            docs.append(d)
    return docs


# Paths for cached per-document embeddings
_DOC_EMB_PATH = Path("data/indices/doc_embs.npy")
_DOC_EMB_META = Path("data/indices/doc_embs_meta.json")


def _build_or_load_doc_embeddings(docs: list):
    """Return normalized per-document embeddings for the provided docs.

    If a cached embeddings file exists and its manifest matches the docs
    ordering/length, load it. Otherwise compute embeddings, normalize them,
    and persist to disk for faster subsequent runs.
    """
    # Basic manifest: hash of each document's text + source to detect changes
    import hashlib
    manifest = []
    texts = [getattr(d, "page_content", "") or "" for d in docs]
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "") if isinstance(d.metadata, dict) else ""
        h = hashlib.sha256()
        h.update(src.encode("utf-8"))
        h.update(b"\x00")
        h.update(texts[i].encode("utf-8"))
        manifest.append(h.hexdigest())
    # try load
    try:
        if _DOC_EMB_PATH.exists() and _DOC_EMB_META.exists():
            existing = json.loads(_DOC_EMB_META.read_text(encoding="utf-8"))
            if isinstance(existing, list) and len(existing) == len(manifest) and existing == manifest:
                embs = np.load(str(_DOC_EMB_PATH))
                # ensure normalized
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
                return (embs / norms).astype("float32")
    except Exception:
        # fallback to recompute
        pass

    # compute embeddings
    # texts variable already built above
    embs = embedder.encode(texts, convert_to_numpy=True)
    embs = np.array(embs, dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    embs = embs / norms

    # persist
    try:
        _DOC_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(_DOC_EMB_PATH), embs)
        _DOC_EMB_META.write_text(json.dumps(manifest), encoding="utf-8")
    except Exception:
        # non-fatal if persist fails
        pass
    return embs


# Key-level embeddings (for keys found in metadata['raw_json'] or first line)
_KEY_EMB_PATH = Path("data/indices/key_embs.npy")
_KEY_EMB_META = Path("data/indices/key_embs_meta.json")


def _build_or_load_key_embeddings(docs: list):
    """Return (embs, meta_list) where meta_list is a list of dicts
    {'doc_idx': int, 'key': str} corresponding to each embedding row.

    The function caches embeddings and a manifest of key hashes so they are
    reused unless document keys change.
    """
    import hashlib
    # Build chunked key candidates: split each key into sliding n-gram chunks
    keys = []
    meta = []
    for i, d in enumerate(docs):
        raw = d.metadata.get("raw_json") if isinstance(d.metadata, dict) else None
        key_candidates = []
        if isinstance(raw, dict) and len(raw) > 0:
            for k in raw.keys():
                ks = str(k).strip()
                if ks:
                    key_candidates.append(ks)
        # Do NOT fall back to page_content; only use explicit JSON keys from
        # metadata['raw_json']. This keeps key embeddings focused on the
        # course code/title and avoids picking up long or noisy values.

        for ks in key_candidates:
            # token-level chunks: 1..4-grams sliding window
            toks = ks.split()
            max_ng = min(4, len(toks))
            seen_chunks = set()
            for n in range(1, max_ng + 1):
                for start in range(0, len(toks) - n + 1):
                    chunk = " ".join(toks[start:start + n]).strip()
                    if chunk and chunk not in seen_chunks:
                        seen_chunks.add(chunk)
                        keys.append(chunk)
                        meta.append({"doc_idx": i, "orig_key": ks, "chunk": chunk})

    if not keys:
        return None, []

    # build manifest as hashes of original keys to detect key-level changes
    # (order-sensitive per-document)
    orig_keys = []
    for i, d in enumerate(docs):
        raw = d.metadata.get("raw_json") if isinstance(d.metadata, dict) else None
        if isinstance(raw, dict) and len(raw) > 0:
            for k in raw.keys():
                orig_keys.append(str(k).strip())
        else:
            pc = (getattr(d, "page_content", "") or "").splitlines()
            if pc:
                orig_keys.append(pc[0].strip())

    manifest = []
    for k in orig_keys:
        h = hashlib.sha256()
        h.update(k.encode("utf-8"))
        manifest.append(h.hexdigest())

    try:
        if _KEY_EMB_PATH.exists() and _KEY_EMB_META.exists():
            existing = json.loads(_KEY_EMB_META.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("manifest") == manifest:
                embs = np.load(str(_KEY_EMB_PATH))
                stored_meta = existing.get("meta", [])
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
                return (embs / norms).astype("float32"), stored_meta
    except Exception:
        pass

    # compute embeddings for key chunks
    embs = embedder.encode(keys, convert_to_numpy=True)
    embs = np.array(embs, dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    embs = embs / norms

    try:
        _KEY_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(_KEY_EMB_PATH), embs)
        _KEY_EMB_META.write_text(json.dumps({"manifest": manifest, "meta": meta}), encoding="utf-8")
    except Exception:
        pass
    return embs, meta


def search_documents(query: str, docs: list, k: int = 6, boost_weight: float = 0.25):
    """Search over the provided Document objects (by page_content).

    Returns a list of dicts: {'doc': Document, 'score': float, 'source': str}.
    Applies a lexical boost if the query contains a course code.
    """
    if not docs:
        return []

    # Use semantic matching against JSON keys only. Build/load key-chunk
    # embeddings and compute similarity between key-chunks and query-chunks.
    key_embs, key_meta = _build_or_load_key_embeddings(docs)
    if key_embs is None or key_embs.shape[0] == 0:
        return []

    # Chunk query into 1..4 token n-grams (same strategy as keys)
    q_toks = query.split()
    q_chunks = []
    max_q_ng = min(4, len(q_toks))
    seen_q = set()
    for n in range(1, max_q_ng + 1):
        for start in range(0, len(q_toks) - n + 1):
            qc = " ".join(q_toks[start:start + n]).strip()
            if qc and qc not in seen_q:
                seen_q.add(qc)
                q_chunks.append(qc)
    if not q_chunks:
        q_chunks = [query]

    # Embed query chunks
    q_embs = embedder.encode(q_chunks, convert_to_numpy=True)
    q_embs = np.array(q_embs, dtype="float32")
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-10)

    # similarity matrix: key_chunk x query_chunk
    sims_mat = key_embs @ q_embs.T
    key_scores = sims_mat.max(axis=1)

    # aggregate per-document by taking the max score among its key-chunks
    sims_doc = np.zeros((len(docs),), dtype="float32")
    sims_doc[:] = -1.0
    for ki, meta in enumerate(key_meta):
        doc_idx = meta.get("doc_idx")
        if doc_idx is None:
            continue
        doc_idx = int(doc_idx)
        sims_doc[doc_idx] = max(sims_doc[doc_idx], float(key_scores[ki]))

    # If no key produced a score (remain -1), set to small value
    sims_doc[sims_doc < 0] = 0.0

    # Exact metadata course_code match gets a very large boost (deterministic)
    m0 = re.search(COURSE_CODE, query, re.I)
    if m0:
        q_variants = generate_course_variants(m0.group(0))
        for i, d in enumerate(docs):
            if isinstance(d.metadata, dict) and d.metadata.get("course_code") in q_variants:
                sims_doc[i] = sims_doc[i] + 10.0

    top_idx = np.argsort(-sims_doc)[:k]
    results = []
    for i in top_idx:
        results.append({"doc": docs[i], "score": float(sims_doc[i]), "source": docs[i].metadata.get("source")})
    return results

    # If the query contains an explicit course code, prefer documents whose
    # `metadata['course_code']` exactly matches any variant of the code.
    m0 = re.search(COURSE_CODE, query, re.I)
    if m0:
        q_variants = generate_course_variants(m0.group(0))
        exact_idxs = [i for i, d in enumerate(docs) if isinstance(d.metadata, dict) and d.metadata.get("course_code") in q_variants]
        if exact_idxs:
            # Strongly prefer exact metadata matches by boosting them high above others
            for i in exact_idxs:
                sims[i] = sims[i] + 10.0

    # semantic key matching: prefer documents whose JSON keys are
    # semantically similar to parts of the query (using the embedder).
    # This helps when the keys are course titles and the user query
    # refers to the course without exact lexical match.
    preferred_doc_idxs = set()
    try:
        key_embs, key_meta = _build_or_load_key_embeddings(docs)
        preferred_doc_idxs = set()
        if key_embs is not None and key_embs.shape[0] > 0:
            # chunk the query into 1..4 token n-grams (same as keys)
            q_toks = query.split()
            q_chunks = []
            max_q_ng = min(4, len(q_toks))
            seen_q = set()
            for n in range(1, max_q_ng + 1):
                for start in range(0, len(q_toks) - n + 1):
                    qc = " ".join(q_toks[start:start + n]).strip()
                    if qc and qc not in seen_q:
                        seen_q.add(qc)
                        q_chunks.append(qc)

            if not q_chunks:
                q_chunks = [query]

            # embed query chunks in a batch
            q_embs = embedder.encode(q_chunks, convert_to_numpy=True)
            q_embs = np.array(q_embs, dtype="float32")
            q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-10)

            # compute key_chunk x query_chunk similarity matrix and take max per key
            sims_mat = key_embs @ q_embs.T
            k_sims = sims_mat.max(axis=1)
            top_key_idx = np.argsort(-k_sims)[:min(12, len(k_sims))]
            for ki in top_key_idx:
                meta = key_meta[ki]
                preferred_doc_idxs.add(meta.get("doc_idx"))
    except Exception:
        preferred_doc_idxs = set()

    # lexical boost for course-code mentions (kept as secondary signal)
    m = re.search(COURSE_CODE, query, re.I)
    if m:
        variants = generate_course_variants(m.group(0))
        for i, d in enumerate(docs):
            text = (getattr(d, "page_content", "") or "").lower()
            meta_text = " ".join(str(v).lower() for v in d.metadata.values()) if isinstance(d.metadata, dict) else ""
            # Check raw_json keys (exact-key match) first, then values/text
            key_exact = False
            raw = d.metadata.get("raw_json") if isinstance(d.metadata, dict) else None
            if isinstance(raw, dict):
                for rk in raw.keys():
                    rk_low = str(rk).lower()
                    if any(re.search(r"\b" + re.escape(v) + r"\b", rk_low) for v in variants):
                        key_exact = True
                        break

            value_contains = any(v in text or v in meta_text for v in variants)

            if key_exact:
                # Strong boost when the original JSON key matches the course code
                sims[i] = sims[i] + (boost_weight * 3.0)
            elif value_contains:
                # Weaker boost when the value/text contains the code
                sims[i] = sims[i] + boost_weight

    # Apply semantic-key preference: strongly boost any doc whose key
    # embedding was among the top key matches for the query.
    try:
        for pd in preferred_doc_idxs:
            if pd is None:
                continue
            # increase sims for that doc index (make sure index in range)
            if 0 <= int(pd) < len(sims):
                sims[int(pd)] = sims[int(pd)] + (boost_weight * 2.5)
    except Exception:
        pass

    top_idx = np.argsort(-sims)[:k]
    results = []
    for i in top_idx:
        results.append({"doc": docs[i], "score": float(sims[i]), "source": docs[i].metadata.get("source")})
    return results


def is_noisy_sentence(s: str) -> bool:
    """Heuristic to detect and filter out noisy navigation/header/footer text
    that sometimes appears in scraped pages (like 'Skip to main content' or
    concatenated navigation tokens).
    """
    if not s:
        return True
    low = s.lower()
    # obvious navigation/footer markers
    noisy_markers = ["skip to main", "site policies", "course descriptions", "last automatic generation", "global footer"]
    for nm in noisy_markers:
        if nm in low:
            return True

    # too few spaces relative to length (likely concatenated words)
    if len(s) > 120 and s.count(" ") < max(2, len(s) // 20):
        return True

    # excessive camelcase-like concatenation
    if re.search(r"[A-Z][a-z]+[A-Z][a-z]+", s):
        return True

    return False

def split_sentences(text: str):
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
    clean = []
    for s in parts:
        # Normalize common non-breaking / zero-width spaces to a regular space
        s = s.replace("\u00A0", " ")
        s = s.replace("\u200B", " ")
        s = s.replace("\u202F", " ")
        s = s.replace("\u205F", " ")
        # Collapse any whitespace run to a single space
        s = re.sub(r"\s+", " ", s)

        # Remove characters that are not typical word/punctuation characters
        # allow spaces and hyphens; place hyphen at the end of the class to avoid
        # defining an accidental range (which caused the previous error)
        s = re.sub(r"[^A-Za-z0-9.,;:/()\s-]", "", s)
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
    """Retrieve top-k sentence-level hits by first searching documents.

    Strategy:
    - Load all Documents (via `load_json`) and perform a document-level
      embedding search with a lexical boost for any course-code variants.
    - From the top document hits, extract sentences, score them, and return
      the highest scoring sentences. This keeps retrieval focused on the
      most relevant documents (reducing wrong-course hits).
    """
    # Load documents and perform a document-level search
    docs = load_all_documents()
    if not docs:
        return []

    # get top documents (w/ lexical boost applied inside search_documents)
    top_docs = search_documents(query, docs, k=8)
    # gather candidate sentences from top docs only
    candidates = []
    for h in top_docs:
        doc = h.get("doc")
        src = h.get("source")
        text = getattr(doc, "page_content", "")
        for s in split_sentences(text):
            if not is_noisy_sentence(s):
                candidates.append((s, src))

    if not candidates:
        return []

    # score sentences with the query embedding
    q_emb = embedder.encode(query)
    scored = [(score_sentence(q_emb, s), s, src) for (s, src) in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])

    # build hits list (dedupe by text+source)
    hits = []
    seen = set()
    for score, s, src in scored:
        key = (s[:160], src)
        if key in seen:
            continue
        seen.add(key)
        hits.append({"text": s, "source": src, "score": float(score)})
        if len(hits) >= k:
            break

    # If the query explicitly contains a course code, ensure sentences that
    # mention that code are included (prepend any missing lexical matches).
    m = re.search(COURSE_CODE, query, re.I)
    if m:
        variants = generate_course_variants(m.group(0))
        lexical_prepend = []
        for s, src in candidates:
            low = s.lower()
            if any(v in low for v in variants):
                key = (s[:160], src)
                if key not in seen:
                    lexical_prepend.append({"text": s, "source": src, "score": 1.0})
                    seen.add(key)
        if lexical_prepend:
            hits = lexical_prepend + hits
            # trim to k
            hits = hits[:k]

    return hits


def two_stage_retrieve(query: str,
                       topk_faiss: int = 80,
                       rerank_top: int = 400,
                       final_k: int = 6,
                       use_cross_encoder: bool = False,
                       key_name_embs=None,
                       key_name_keys=None):
    """Two-stage retrieval using FAISS for recall + sentence reranking.

    Steps:
    1) Recall top chunks from the prebuilt FAISS `index` (uses `meta` for text).
    2) Extract and filter sentences from those chunks.
    3) Rerank candidate sentences using the bi-encoder (cosine) or an
       optional CrossEncoder.

    Returns a list of hits: {'text': sentence, 'source': source, 'score': float}.
    """
    # embed the query (bi-encoder)
    q_emb = embedder.encode(query, convert_to_numpy=True)
    q_emb = np.array(q_emb, dtype="float32")
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

    # Prefer JSON-key-only search when chunks were created from processed
    # JSON entries: collect unique `orig_key` values from `meta` and embed
    # those keys. This ensures queries are matched against canonical JSON
    # keys (course codes/titles) instead of arbitrary chunk text.
    key_to_indices = {}
    for ci, chunk in enumerate(meta):
        ok = chunk.get("orig_key")
        if ok:
            key_to_indices.setdefault(ok, []).append(ci)

    idxs = []
    try:
        if key_to_indices:
            # If the user explicitly mentioned a course code (e.g. "CS160",
            # "CICS 160"), perform a deterministic lexical lookup first.
            # This prefers chunks whose `course_code` metadata or whose
            # `orig_key` contains the course code variants. If we find any
            # such chunks, use them as the primary recall set and skip the
            # semantic-key ranking step. This prevents value-only matches
            # (where "160" appears in a prereq list) from outranking the
            # actual course entry.
            m_code = re.search(COURSE_CODE, query, re.I)
            if m_code:
                variants = generate_course_variants(m_code.group(0))
                matched_idxs = []
                for ci, ch in enumerate(meta):
                    # prefer explicit `course_code` if present on the chunk
                    cc = ch.get("course_code")
                    if isinstance(cc, str) and cc in variants:
                        matched_idxs.append(ci)
                        continue
                    # fall back to checking orig_key textual match
                    ok = (ch.get("orig_key") or "").lower()
                    for v in variants:
                        # whole-word check against the orig_key string
                        if re.search(r"\b" + re.escape(v) + r"\b", ok):
                            matched_idxs.append(ci)
                            break
                if matched_idxs:
                    # preserve order and cap to topk_faiss
                    seen_local = set()
                    ordered = []
                    for i in matched_idxs:
                        if i not in seen_local:
                            ordered.append(i)
                            seen_local.add(i)
                        if len(ordered) >= topk_faiss:
                            break
                    idxs = ordered
                    # skip the rest of the key-embedding path
                    # proceed directly to collecting sentences from these chunks
                    k_embs = None
                    k_keys = []
                else:
                    k_embs = None
                    k_keys = []

            # Prefer using persisted full-key embeddings if provided by caller
            keys = list(key_to_indices.keys())
            k_embs = None
            k_keys = []
            if key_name_embs is not None and key_name_keys is not None:
                # build arrays for the intersection of available persisted keys
                persist_map = {k: i for i, k in enumerate(key_name_keys)}
                present = [k for k in keys if k in persist_map]
                absent = [k for k in keys if k not in persist_map]
                parts = []
                if present:
                    idxs_for_present = [persist_map[k] for k in present]
                    parts.append(np.array(key_name_embs, dtype="float32")[idxs_for_present])
                if absent:
                    # embed missing keys on demand
                    a_embs = embedder.encode(absent, convert_to_numpy=True)
                    a_embs = np.array(a_embs, dtype="float32")
                    a_embs = a_embs / (np.linalg.norm(a_embs, axis=1, keepdims=True) + 1e-10)
                    parts.append(a_embs)
                if parts:
                    k_embs = np.vstack(parts)
                    k_keys = present + absent
            else:
                # no persisted embeddings passed; embed all keys now
                k_keys = keys
                k_embs = embedder.encode(k_keys, convert_to_numpy=True)
                k_embs = np.array(k_embs, dtype="float32")
                k_embs = k_embs / (np.linalg.norm(k_embs, axis=1, keepdims=True) + 1e-10)

            if (k_embs is None or k_embs.shape[0] == 0) and not idxs:
                # fallback to FAISS search over chunk vectors (only if we
                # haven't already selected lexical-matched chunk indices)
                D, I = index.search(q_emb.reshape(1, -1), topk_faiss)
                idxs = I[0].tolist()
            else:
                # If k_embs is available, score keys semantically and expand
                # to their chunk indices. Otherwise, fall back to FAISS over
                # chunk vectors (but only if idxs is still empty).
                if k_embs is not None and getattr(k_embs, "shape", None) and k_embs.shape[0] > 0:
                    sims = (k_embs @ q_emb.reshape(-1, 1)).reshape(-1)
                    topk = min(topk_faiss, len(sims))
                    top_key_idx = np.argsort(-sims)[:topk]
                    for ki in top_key_idx:
                        key = k_keys[int(ki)]
                        for ci in key_to_indices.get(key, []):
                            idxs.append(int(ci))
                    idxs = idxs[:topk_faiss]
                elif not idxs:
                    D, I = index.search(q_emb.reshape(1, -1), topk_faiss)
                    idxs = I[0].tolist()
        else:
            # fallback to FAISS search over chunk vectors
            D, I = index.search(q_emb.reshape(1, -1), topk_faiss)
            idxs = I[0].tolist()
    except Exception:
        return []
    candidates = []
    seen = set()
    for idx in idxs:
        if idx < 0 or idx >= len(meta):
            continue
        chunk = meta[idx]
        text = chunk.get("text", "")
        src = chunk.get("source") or chunk.get("chunk_id")
        for s in split_sentences(text):
            if is_noisy_sentence(s):
                continue
            key = (s[:160], src)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((s, src))

    if not candidates:
        return []

    # limit rerank pool for efficiency
    if len(candidates) > rerank_top:
        candidates = candidates[:rerank_top]

    sentences = [s for s, _ in candidates]
    sources = [src for _, src in candidates]

    reranked = []
    if use_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder
            cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[query, s] for s in sentences]
            scores = cross.predict(pairs)
            for score, s, src in zip(scores, sentences, sources):
                reranked.append((float(score), s, src))
        except Exception:
            # fall back to bi-encoder if cross-encoder not available
            use_cross_encoder = False

    if not use_cross_encoder:
        s_embs = embedder.encode(sentences, convert_to_numpy=True)
        s_embs = np.array(s_embs, dtype="float32")
        norms = np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-10
        s_embs = s_embs / norms
        qn = q_emb.reshape(1, -1)
        sims = (s_embs @ qn.T).reshape(-1)
        for sim, s, src in zip(sims.tolist(), sentences, sources):
            reranked.append((float(sim), s, src))

    # sort descending
    reranked.sort(key=lambda x: -x[0])

    # dedupe and return top-k
    hits = []
    seen2 = set()
    for score, s, src in reranked:
        key = (s[:160], src)
        if key in seen2:
            continue
        seen2.add(key)
        hits.append({"text": s, "source": src, "score": float(score)})
        if len(hits) >= final_k:
            break

    # If explicit course code in query, ensure lexical matches are present
    m = re.search(COURSE_CODE, query, re.I)
    if m:
        variants = generate_course_variants(m.group(0))
        lexical_prepend = []
        for s, src in candidates:
            low = s.lower()
            if any(v in low for v in variants):
                key = (s[:160], src)
                if key not in seen2:
                    lexical_prepend.append({"text": s, "source": src, "score": 1.0})
                    seen2.add(key)
        if lexical_prepend:
            hits = lexical_prepend + hits
            hits = hits[:final_k]

    return hits

def retrieve_context(query: str, k_docs: int = 8, top_k_sentences: int = 5):
    """Retrieve relevant sentences and their sources for `query`.

    Returns a list of dicts with keys: 'score', 'sentence', 'source'. Does NOT
    generate a final summarized answer — only retrieval.
    """
    hits = retrieve(query, k=k_docs)
    q_emb = embedder.encode(query)

    # collecting candidate sentences from retrieved documents, filtering noisy ones
    candidates = []
    for h in hits:
        for s in split_sentences(h["text"]):
            if len(s) > 20 and not is_noisy_sentence(s):
                candidates.append((s, h["source"]))

    if not candidates:
        # return empty list and the set of sources we examined
        return [], sorted(list({h["source"] for h in hits}))

    # scoring & picking top
    scored = [(score_sentence(q_emb, s), s, src) for (s, src) in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: top_k_sentences * 3]  # take a slightly larger pool

    # dedupe while preserving order and limit to top_k_sentences
    results = []
    seen = set()
    for score, s, src in top:
        key = (s[:160], src)
        if key in seen:
            continue
        seen.add(key)
        results.append({"score": float(score), "sentence": s, "source": src})
        if len(results) >= top_k_sentences:
            break

    return results


def retrieve_only(query: str, k_docs: int = 8, top_k_sentences: int = 5):
    """Convenience wrapper that returns (results, sources) where results is a
    list of {score, sentence, source} and sources is a sorted list of unique
    source ids examined. This mirrors the previous API shape somewhat but
    avoids generating a full textual answer.
    """
    results = retrieve_context(query, k_docs=k_docs, top_k_sentences=top_k_sentences)
    if not results:
        # no sentence results; return sources examined so caller knows what was searched
        hits = retrieve(query, k=k_docs)
        sources = sorted(list({h["source"] for h in hits}))
        return [], sources
    sources_set = set()
    for r in results:
        if isinstance(r, dict):
            src = r.get("source")
            if src is not None:
                sources_set.add(src)
    sources = sorted(list(sources_set))
    return results, sources


def generate_answer(query: str,
                    k_docs: int = 8,
                    top_k_sentences: int = 6,
                    llm_model: str = "gpt-3.5-turbo",
                    key_name_embs=None,
                    key_name_keys=None,
                    force_llm: bool = False):
    """Generate a full answer to `query` using retrieved context.

    Tries to use OpenAI's chat completion if `openai` is available and
    OPENAI_API_KEY is set; otherwise falls back to a deterministic
    concatenation of retrieved sentences (with citations).

    Returns (answer_text, sources_list)
    """
    # Use two-stage FAISS recall + rerank for retrieval when generating answers
    topk_faiss = max(40, k_docs * 10)
    rerank_top = max(200, k_docs * 50)
    results = two_stage_retrieve(query, topk_faiss=topk_faiss, rerank_top=rerank_top, final_k=top_k_sentences,
                                  key_name_embs=key_name_embs, key_name_keys=key_name_keys)

    if not results:
        return "I couldn't find relevant information in the index.", []

    # Quick heuristic: if any retrieved sentence explicitly mentions prerequisites
    # or contains the course code (including interchangeable prefixes), return a
    # deterministic extractive answer so we don't rely on the LLM to decide
    # sufficiency.
    m = re.search(COURSE_CODE, query, re.I)
    query_variants = generate_course_variants(m.group(0)) if m else []

    prereq_matches = []
    for r in results:
        if not isinstance(r, dict):
            continue
        sent = r.get("text", "")
        sent_low = sent.lower()
        # match prereq keywords or explicit course code mentions or any variant
        has_course_variant = any(v in sent_low for v in query_variants) if query_variants else bool(re.search(COURSE_CODE, sent, re.I))
        if PREREQ_KEYS.search(sent) or has_course_variant:
            prereq_matches.append((sent, r.get("source")))

    if prereq_matches and not force_llm:
        # Build a short extractive answer from the matched sentences. Keep
        # this deterministic (no-LLM) path concise: dedupe, limit to a few
        # important sentences and return sources. If the caller sets
        # `force_llm=True`, skip this and let the LLM produce a natural
        # language response instead.
        lines = []
        sources_set = set()
        seen_sent = set()
        for s, src in prereq_matches:
            s_norm = s.strip()
            if not s_norm or s_norm in seen_sent:
                continue
            seen_sent.add(s_norm)
            lines.append(s_norm)
            if src:
                sources_set.add(src)
            if len(lines) >= 3:
                break
        answer_text = "\n".join(lines)
        sources_sorted = sorted(list(sources_set))
        return answer_text, sources_sorted

    # Build context block for the prompt
    context_lines = []
    for i, r in enumerate(results, start=1):
        src = r.get("source", "") if isinstance(r, dict) else ""
        sent = r.get("text", "") if isinstance(r, dict) else str(r)
        context_lines.append(f"[{i}] ({src}) {sent}")
    context_text = "\n".join(context_lines)

    # Short prompt instructing the model to answer based on the retrieved
    # snippets and to cite sources in-line using the bracket numbers above.
    prompt = (
        f"You are an assistant that answers user questions using only the provided "
        f"context. If the context lacks enough information, say so instead of "
        f"making things up. Use inline citations referencing the bracketed items "
        f"from the context (for example: (see [1])).\n\n"
        f"Context snippets:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely and cite which context snippets you used.")

    # Use OpenAI chat completion if configured; otherwise fall back.
    if client is not None:
        try:
            resp = client.chat.completions.create(model=llm_model,
                                                  messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                            {"role": "user", "content": prompt}],
                                                  temperature=0,
                                                  max_tokens=512)
            # Safely extract the text from the completion response
            text = ""
            try:
                # newer SDK may expose choices -> message -> content
                candidate = None
                if hasattr(resp, "choices") and len(resp.choices) > 0:
                    # try structured access
                    try:
                        candidate = resp.choices[0].message.content
                    except Exception:
                        # fallback to older shape
                        candidate = getattr(resp.choices[0], "text", None)
                if candidate is None:
                    text = ""
                else:
                    text = str(candidate).strip()
            except Exception:
                # Fallback: stringify the whole response
                text = str(resp)
            # sources: collect unique non-None sources from results
            sources_set = set()
            for r in results:
                if isinstance(r, dict):
                    s = r.get("source")
                    if s is not None:
                        sources_set.add(s)
            sources = sorted(list(sources_set))
            return text, sources
        except Exception as e:
            # fallback to deterministic concatenation below; show reason
            print("OpenAI call failed; falling back to concatenation:", repr(e))
    else:
        # no API key/client configured
        print("OpenAI client not configured (OPENAI_API_KEY missing); falling back to deterministic concatenation.")

    # Fallback: deterministic answer by combining retrieved sentences.
    # Use 'text' (two-stage retriever) or 'sentence' (older helpers) keys.
    lines = []
    for r in results:
        if isinstance(r, dict):
            l = r.get("text") or r.get("sentence") or ""
        else:
            l = str(r)
        if l:
            lines.append(l)
    # build sources set
    sources_set = set()
    for r in results:
        if isinstance(r, dict):
            s = r.get("source")
            if s is not None:
                sources_set.add(s)
    sources_sorted = sorted(list(sources_set))
    answer = (
        "Retrieved information (no LLM available):\n"
        + "\n".join(f"- {l}" for l in lines)
        + "\n\nSources: "
        + ", ".join(sources_sorted)
    )
    return answer, sources_sorted

if __name__ == "__main__":
    q = input("Ask a question: ")
    # Run the generative pipeline (uses OpenAI if available, otherwise falls
    # back to a deterministic concatenation of retrieved snippets).
    answer_text, sources = generate_answer(q, k_docs=8, top_k_sentences=6)
    # results, _ = retrieve_only(q, k_docs=8, top_k_sentences=6)
    # print("=== Retrieved Sentences ===\n")
    # for r in results:
    #     if isinstance(r, dict):
    #         print(f"- ({r.get('source', '')}) {r.get('sentence', '')}")
    #     else:
    #         print(f"- {str(r)}")
    print("\n=== Generated Answer ===\n")
    print(answer_text)
    if sources:
        print("\nSources:", ", ".join(sources))
