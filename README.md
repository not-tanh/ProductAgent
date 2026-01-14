# How to run

## Prerequisites
- Python **3.12+**
- Docker (for Qdrant, Redis, optional self-host Langfuse)
- Internet access if you want to use the web analysis tool

Repository layout (high level):
- `agents/` — single-agent implementation
- `multiagents/` — multi-agent (LangGraph) implementation
- `data_pipeline/` — cleaning → duckdb → qdrant ingestion
- `ui/` — Streamlit chat UI

---

## 1. Run Qdrant

The ingestion pipeline and internal search use Qdrant via:
- `QDRANT_URL` (e.g. `http://localhost:6333`)
- `COLLECTION_NAME` (e.g. `products`)

Start Qdrant with Docker:

```bash
docker pull qdrant/qdrant

docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

---

## 2. Run Redis

Redis is used for:
- storing chat history per session

```bash
docker run -d --name redis -p 6379:6379 redis
```

---

## 3. Install necessary libraries

### Install with `uv`

From repo root:

```bash
uv venv
source .venv/bin/activate
uv sync
```

---

## 4. Check configuration

Copy `.env.example` file in the repo root and rename the copied file to `.env`. Most modules call `load_dotenv()`.

### 4.1 Core services

```env
# Qdrant
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=products

# Redis
REDIS_URL=redis://localhost:6379/
SESSION_TTL_SECONDS=86400
SESSION_LOCK_TTL_SECONDS=10
MAX_HISTORY_MESSAGES=10
SESSION_KEY_PREFIX=chat:session:
LOCK_KEY_PREFIX=chat:lock:
```

### 4.2 Models

This code expects OpenAI model names via environment variables (e.g: gpt-5.2, gpt-5-nano, etc.):

```env
ORCHESTRATOR_MODEL=...
PRODUCT_SEARCH_MODEL=...
WEB_ANALYSIS_MODEL=...
PRODUCT_ANALYSIS_MODEL=...
```

### 4.3 Embedding models (used by Qdrant ingestion + internal search)

```env
DENSE_MODEL=...
SPARSE_MODEL=...
```

### 4.4 DuckDB (used by product analysis tool)

The analysis tool opens DuckDB using `DUCKDB`:

```env
DUCKDB=./products.duckdb
```

(Adjust the path if you write the DuckDB file elsewhere.)

### 4.5 Optional: ingestion tuning

```env
EMBED_BATCH=8
UPLOAD_BATCH=1024
SHARDS=2
```

---

## 5. Run data pipeline

There are 3 steps:
1) download + clean data → `products.parquet`
2) build `products.duckdb` for SQL analytics
3) ingest embeddings + payload into Qdrant

The repo includes `run_data_pipeline.sh`:
```bash
bash run_data_pipeline.sh
```

## 6. Run Langfuse (cloud or self-host)

Set these in `.env`:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

---

## 7. Run agent API

You can run either the single-agent API or the multi-agent (LangGraph) API.

### 7.1 Single-agent API

From repo root:

```bash
uvicorn agents.main:app --port 8000
```

### 7.2 Multi-agent (LangGraph) API

Recommended run command (to avoid import-path ambiguity):

```bash
cd multiagents
uvicorn main:app --port 8000
```

---

## 8. Run chat UI (Streamlit)

The Streamlit UI is in `ui/main.py` and calls:
- `CHAT_API_BASE_URL` (default: `http://localhost:8000`)
- endpoint: `/chat`
- header: `X-Session-Id`

Run:

```bash
streamlit run ui/main.py --server.port 8501
```

Open in browser:
- `http://localhost:8501`
