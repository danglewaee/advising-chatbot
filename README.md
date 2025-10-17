# Academic Advising Chatbot (RAG)

This project builds an academic advising chatbot for UMass Amherst.
It uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about course prerequisites, degree requirements, and academic policies.

## Architecture

- **Data collection:** `pipelines/download_pages.py` + `bulk_sources_crawler.py`
- **Chunking:** `pipelines/chunk.py`
- **Embedding & Indexing:** `pipelines/embed_index.py` (local SentenceTransformers)
- **Retrieval:** `rag/pipeline.py`
- **Web Interface:** `apps/api/main.py` (FastAPI web UI)


## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
