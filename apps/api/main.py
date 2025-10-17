from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from rag.pipeline import answer

app = FastAPI(title="Advising Chatbot RAG API")
templates = Jinja2Templates(directory="apps/api/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask")
def ask(q: str = Query(..., description="Your question")):
    return {"question": q, "answer": answer(q)}
