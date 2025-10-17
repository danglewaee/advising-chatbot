from pathlib import Path
import json, re

RAW = Path("data/raw")
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
for p in RAW.glob("*.txt"):
    src = p.stem
    txt = normalize(p.read_text(encoding="utf-8", errors="ignore"))
    for j, ch in enumerate(chunk_text(txt)):
        chunks.append({"chunk_id": f"{src}-{j}", "source": src, "text": ch})

OUT.joinpath("chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ wrote {len(chunks)} chunks -> data/processed/chunks.json")
