from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import numpy as np
from pathlib import Path
from rag.pipeline import generate_answer as answer
import uvicorn

app = FastAPI(title="Advising Chatbot RAG API")
templates = Jinja2Templates(directory="apps/api/templates")


@app.on_event("startup")
def load_key_name_embeddings():
    idx_dir = Path("data/indices")
    emb_path = idx_dir / "key_name_embs.npy"
    meta_path = idx_dir / "key_name_meta.json"
    if emb_path.exists() and meta_path.exists():
        try:
            app.state.key_name_embs = np.load(str(emb_path))
            app.state.key_name_keys = json.loads(meta_path.read_text(encoding="utf-8")).get("keys", [])
            print(f"Loaded {len(app.state.key_name_keys)} key-name embeddings")
        except Exception as e:
            print("Failed to load key-name embeddings:", e)
            app.state.key_name_embs = None
            app.state.key_name_keys = None
    else:
        app.state.key_name_embs = None
        app.state.key_name_keys = None

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask")
def ask(q: str = Query(..., description="Your question")):
    # Pass preloaded key-name embeddings into the generate_answer function
    key_embs = getattr(app.state, "key_name_embs", None)
    key_keys = getattr(app.state, "key_name_keys", None)
    ans = answer(q, key_name_embs=key_embs, key_name_keys=key_keys)
    return {"question": q, "answer": ans}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)