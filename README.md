# RAG Assistant

This repository contains a full-stack RAG (Retrieval-Augmented Generation) application
that allows users to upload documents and interact with their content through an
intelligent conversational assistant powered by local and cloud LLMs.

Unlike basic chatbots, this system focuses on **source-grounded answers**, **zero
hallucination policy** (explicit refusal when no relevant content is found), and
**multi-LLM flexibility** (Ollama local + Groq cloud).

---

## Environment Setup (Requirements)

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install

# Or run everything with Docker
docker-compose up --build
```

---

---

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline for document chat.

Instead of generating answers blindly, the chatbot:

1. Ingests uploaded documents (PDF, DOCX) into a vector database
2. Retrieves the most relevant chunks using semantic search
3. Uses that context to answer user questions with source-grounded responses

This ensures accurate, explainable, and document-grounded responses.

---

## Key Features

- Semantic document search using ChromaDB
- Document-aware responses (RAG retrieval)
- FastAPI backend
- Sentence-Transformers embeddings
- Multi-LLM support — Ollama (local) / Groq (cloud)
- User authentication (JWT)
- React frontend (Vite + Tailwind)
- Docker Compose deployment
- Evaluation pipeline (Benchmark.json)

---
## Project Structure

```
RAG_Assistant/
├── backend/
│   ├── app.py              # FastAPI app entry point
│   ├── auth.py             # JWT authentication
│   ├── config.py           # Configuration & env variables
│   ├── ingest.py           # Document ingestion pipeline
│   ├── rag.py              # RAG pipeline (retrieval + generation)
│   ├── memory.py           # Conversation memory management
│   ├── llm_providers.py    # Ollama & Groq LLM abstraction
│   ├── eval_metrics.py     # Evaluation metrics
│   ├── run_evaluation.py   # Benchmark runner
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.jsx
│   │   │   ├── Header.jsx
│   │   │   └── Message.jsx
│   │   ├── pages/
│   │   │   ├── Chat.jsx
│   │   │   ├── Documents.jsx
│   │   │   └── Login.jsx
│   │   └── services/
│   │       └── api.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── Dockerfile
├── Benchmark.json          # Evaluation dataset
├── Benchmark2.json         # Second evaluation dataset
├── docker-compose.yml
└── memory.db               # SQLite conversation memory
```
---

## How It Works (RAG Flow)

1. User uploads a document (PDF or DOCX)
2. Document is chunked and converted into embeddings
3. Embeddings are stored in ChromaDB vector database
4. User sends a question → question is vectorized
5. Top relevant chunks are retrieved and reranked
6. Retrieved chunks are injected as context into the LLM prompt
7. The chatbot answers **based on the uploaded documents**

---
## Installation

**1. Clone the repository**

```bash
git clone https://github.com/ElMariem/RAG_Assistant
cd RAG_Assistant
```

**2. Add your API keys**

Create a `.env` file at the root:

```env
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=your_jwt_secret
```

**3. Run with Docker**

```bash
docker-compose up --build
```

App available at:

http://localhost:8000/docs — Swagger UI

http://localhost:8080 — Frontend

http://localhost:8000 — Backend API

---

## Technologies Used

- Python 3.11
- FastAPI
- ChromaDB
- Sentence-Transformers
- Ollama / Groq
- React + Vite (Frontend)
- Tailwind CSS
- Docker / Docker Compose
- JWT (Authentication)

---

## Evaluation

```bash
docker exec -it rag_backend python run_evaluation.py
```

Results saved in `eval_results/`. Benchmark datasets contain question-answer pairs
to measure retrieval precision and generation faithfulness.

---

## Notes

- Large files (`chroma_db/`, `memory.db`) are intentionally excluded from Git
- Vector store can be rebuilt locally by re-ingesting documents
- Ollama / LLM usage is optional and modular — switch between local and cloud at runtime

---
