# RAG Assistant

A full-stack, **source-grounded AI document assistant** powered by Retrieval-Augmented Generation (RAG).
Upload PDF/DOCX documents and chat with their content through an intelligent conversational assistant
running on **local (Ollama) or cloud (Groq) LLMs**.

Unlike basic chatbots, this system follows a strict source-grounded answering policy, refusing to answer when the retrieved documents do not contain sufficient information.

## Key Features

- **Multimodal document ingestion**: extracts and indexes text, tables, and diagrams
- **Hybrid retrieval pipeline**: semantic search followed by CrossEncoder reranking
- **Source-grounded generation**: answers are restricted to retrieved document content
- **Multi-LLM support**: switch between local Ollama and cloud Groq models
- **Conversation memory**: persistent, per-user conversation history
- **Streaming responses**: real-time token streaming via SSE
- **User authentication**: JWT-based authentication with per-user document isolation
- **Evaluation pipeline**: reproducible benchmark for retrieval and generation quality
- **Docker deployment**: containerized backend and frontend with Docker Compose

## Architecture

```mermaid
flowchart TD
    A[React Frontend&lt;br/&gt;Vite + Tailwind :8080] --&gt;|POST /api/chat, /api/documents/*| B[FastAPI Backend :8000]
    B --&gt; C[JWT Auth]
    B --&gt; D[Ingestion Pipeline]
    B --&gt; E[RAG Pipeline&lt;br/&gt;retrieve + rerank + generate]
    D --&gt; F[(ChromaDB&lt;br/&gt;vector store)]
    E --&gt; F
    E --&gt; G[(SQL Server<br/>users, conversations)]
    E --&gt; H{{LLM Provider}}
    H --&gt;|local| I[Ollama]
    H --&gt;|cloud| J[Groq]
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS, react-router-dom |
| Backend | Python 3.11, FastAPI, Uvicorn |
| Auth | JWT (PyJWT, HS256), bcrypt |
| Vector Store | ChromaDB (persistent, cosine distance) |
| Embeddings | Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Reranking | CrossEncoder (mmarco-mMiniLMv2-L12-H384-v1) |
| LLMs (local) | Ollama — qwen2.5vl:7b (multimodal generation) |
| LLMs (cloud) | Groq — openai/gpt-oss-20b, qwen/qwen3-32b |
| Document Parsing | Docling (structure), PyMuPDF (page rendering/cropping) |
| Memory | SQL Server |
| LLM Judge (eval) | openai/gpt-oss-120b |
| Deployment | Docker, Docker Compose |

## Project Structure

```
RAG_Assistant/
├── backend/
│   ├── app.py              # FastAPI app entry point
│   ├── auth.py             # JWT authentication
│   ├── config.py           # Configuration & env variables
│   ├── ingest.py           # Multimodal ingestion (text, tables, diagrams)
│   ├── rag.py              # RAG pipeline (retrieval, reranking, generation)
│   ├── memory.py           # Conversation memory management
│   ├── llm_providers.py    # Ollama & Groq LLM abstraction
│   ├── eval_metrics.py     # Evaluation metrics
│   ├── run_evaluation.py   # Benchmark runner
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/     # ChatBox, Header, Message
│   │   ├── pages/          # Chat, Documents, Login
│   │   └── services/       # api.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── Dockerfile
├── Benchmark.json          # Evaluation dataset (35 annotated cases)
├── Benchmark2.json         # Second evaluation dataset
├── docker-compose.yml
└── memory.db             
```

## How It Works

1. **Ingestion** — a PDF/DOCX is parsed by Docling into sections, tables, and figures.
   Text is chunked by token count (with overlap), tables become sentence-form chunks, and diagrams are
   cropped and captioned by a vision model using the surrounding page text as context.
2. **Indexing** — every chunk is embedded (Sentence-Transformers) and stored in ChromaDB with rich
   metadata: source file, page, section path, table structure, image path.
3. **Retrieval** — the question is embedded and the top-40 chunks are fetched, then reranked by a
   CrossEncoder with a per-file diversity cap down to 10 chunks, and fitted into the context window budget.
4. **Generation** — the reranked chunks + recent conversation history are injected into a strict
   system prompt ("answer ONLY from the context"), and the answer streams back with its source
   references.

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/ElMariem/RAG_Assistant
cd RAG_Assistant
```

**2. Add your API keys**

Create a `.env` file at the root:

```env
GROQ_API_KEY=your_groq_api_key   # optional
SECRET_KEY=your_jwt_secret
DATABASE_URL=your_sqlserver_odbc_connection_string
```

**3. Run with Docker**

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8080 |
| Backend API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

&gt; Local LLM mode requires [Ollama](https://ollama.com) running with a multimodal model
&gt; (`ollama pull qwen2.5vl:7b`) and `LLM_BACKEND=ollama` in `.env`.

**Or run manually:**

```bash
# Backend
cd backend && pip install -r requirements.txt && uvicorn app:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

## Evaluation

Benchmark of **35 expert-annotated test cases** (5 categories: factual single-chunk, table, multi-source,
unanswerable, diagram) run against a research corpus on Alzheimer's detection from EEG signals.
Retrieval metrics are identical for both engines (same embeddings + reranker); generation was judged by
an LLM judge (`openai/gpt-oss-120b`).

```bash
docker exec -it rag_backend python run_evaluation.py
```

### Retrieval metrics (identical for Ollama and Groq)

| Metric | Value |
|---|---|
| Hit rate @ pool (10 chunks) | **89.7%** |
| Recall @ pool (10 chunks) | **89.7%** |
| Recall @ fixed k=5 | **87.9%** |
| MRR (mean reciprocal rank) | **0.633** |
| Precision @ fixed k=5 | 0.149 |

*Precision is structurally low: most questions have only 1–2 annotated relevant sources, so a fixed
window of 5 chunks cannot score high even with perfect retrieval confirmed by the high recall.*

### Generation metrics — Ollama (local) vs Groq (cloud)

| Metric | Ollama (local) | Groq (cloud) |
|---|---|---|
| Strict accuracy | 85.7% (30/35) | **97.1% (34/35)** |
| Correctness | 87.1% | **98.6%** |
| Faithfulness | 85.7% | **94.3%** |
| Hallucination rate | 14.3% | **5.7%** |
| Answer relevancy | **95.4%** | 94.9% |

### Accuracy / faithfulness by question category

| Category | n | Ollama | Groq |
|---|---|---|---|
| Factual (single chunk) | 12 | 1.00 / 1.00 | 1.00 / 1.00 |
| Table | 6 | 0.83 / 0.83 | 1.00 / 1.00 |
| Multi-source | 6 | 0.50 / 0.50 | 0.83 / 0.83 |
| Unanswerable (refusal) | 6 | 1.00 / 1.00 | 1.00 / 1.00 |
| Diagram | 5 | 0.80 / 0.80 | 1.00 / 0.80 |

*Multi-source questions remain the hardest task for both engines. Diagram faithfulness remains at
0.80 with Groq; the main limitation is the quality of the vision-generated descriptions produced
during ingestion rather than the generation engine itself.*

## Notes

- Large files (`chroma_db/`, `memory.db`, extracted figures) are intentionally excluded from Git
  the vector store is rebuilt locally by re-ingesting documents.
- SQL Server connection (ODBC) is configured via the `.env` file.
- Ollama / Groq usage is optional and modular switch between local and cloud at runtime.
