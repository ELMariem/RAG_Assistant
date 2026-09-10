# RAG Assistant — Intelligent Document Chat System

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

## Architecture Overview

The pipeline employs a three-layer RAG architecture designed to ensure retrieval
precision and generation faithfulness:

1. **Ingestion Layer** — Raw documents (PDF, DOCX) are chunked, embedded using
   Sentence Transformers, and indexed into ChromaDB.

2. **Retrieval & Generation Layer** — User queries are vectorized and compared
   to indexed fragments:
   - **Top 40** fragments retrieved by cosine similarity
   - **Reranked** by cross-encoder and diversified by source (top 10 kept)
   - **Prompt** constructed from fragments + conversation history + question
   - **LLM** (Ollama or Groq) generates a streamed, source-anchored response

3. **Applicative Layer** — React frontend communicates with the FastAPI backend
   via REST API, with JWT-based authentication per user.
