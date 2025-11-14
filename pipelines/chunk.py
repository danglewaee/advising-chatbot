from pathlib import Path
import json, re

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def normalize(txt: str) -> str:
    txt = re.sub(r"\r", "", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def chunk_text(text, size=400, overlap=80):
    words = text.split()
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        yield " ".join(words[i:i+size])

chunks = []
# Only read processed JSON files. The JSON directory contains the
# formatted course keys and descriptions that are the canonical source
# for course metadata; the raw .txt files are redundant.
JSON_DIR = OUT.joinpath("json")
if not JSON_DIR.exists():
    print(f"Warning: {JSON_DIR} does not exist; no chunks created")
else:
    for jf in JSON_DIR.glob("*.json"):
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        src = jf.stem
        # obj is expected to be a mapping: key -> value (value may be list/string/dict)
        for k, v in obj.items():
            # build a readable text where the key is on the first line
            if isinstance(v, list):
                body_lines = []
                for item in v:
                    if isinstance(item, (dict, list)):
                        body_lines.append(json.dumps(item, ensure_ascii=False))
                    else:
                        body_lines.append(str(item))
                text = f"{k}\n" + "\n".join(body_lines)
            elif isinstance(v, dict):
                text = f"{k}\n" + json.dumps(v, ensure_ascii=False)
            else:
                text = f"{k}\n" + str(v)

            text = normalize(text)

            # try to extract a canonical course code for this key/value so
            # retrieval can prefer chunks that are actually the course entry
            # (e.g. 'CICS 160'). We store a lowercase normalized form like
            # 'cics 160' in `course_code` when present.
            course_code = None
            m = re.search(r"\b(CICS|COMPSCI|INFO|MATH|STAT|PHYS|CS)\s?-?(\d{2,3}[A-Z]?)\b", k, re.I)
            if not m:
                # try the value text as a fallback
                m = re.search(r"\b(CICS|COMPSCI|INFO|MATH|STAT|PHYS|CS)\s?-?(\d{2,3}[A-Z]?)\b", text, re.I)
            if m:
                course_code = f"{m.group(1)} {m.group(2)}".lower()

            # chunk the combined key+value text as with raw files
            for j, ch in enumerate(chunk_text(text)):
                # create a compact, safe chunk id using the file stem and a
                # sanitized key prefix to help debugging
                safe_key = re.sub(r"[^A-Za-z0-9]+", "_", k).strip("_")[:40]
                chunk_id = f"{src}-json-{safe_key}-{j}"
                chunks.append({"chunk_id": chunk_id, "source": src, "text": ch, "orig_key": k, "course_code": course_code})

# Also include processed JSON entries (course keys / descriptions) so the
# FAISS index contains canonical course codes and titles. We create at least
# one chunk per JSON key that begins with the key on the first line so
# lexical queries like "CICS 160" will be indexed.

OUT.joinpath("chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ wrote {len(chunks)} chunks -> data/processed/chunks.json")
