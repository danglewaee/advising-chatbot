from fastapi import FastAPI, Query
from rag.pipeline import answer

app = FastAPI(title="Advising Chatbot RAG API")

@app.get("/ask")
def ask(q: str = Query(..., description="Your question")):
    return {"question": q, "answer": answer(q)}

# chạy local:
# uvicorn apps.api.main:app --reload
