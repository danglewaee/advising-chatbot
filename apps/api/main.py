from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from rag.pipeline import answer
import uvicorn

app = FastAPI(title="Advising Chatbot RAG API")
templates = Jinja2Templates(directory="apps/api/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask")
def ask(q: str = Query(..., description="Your question")):
    return {"question": q, "answer": answer(q)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)