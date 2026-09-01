"""
RAG chain: given a user question, retrieve relevant chunks from ChromaDB,
then ask the generator model to answer using that context."""

import base64
import os
import config
import llm_providers
import logging
from sentence_transformers import CrossEncoder
import numpy as np
from ingest import _tokenizer

logger = logging.getLogger(__name__)
_reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
def encode_image_base64(path: str) -> str:
    """Read an image file from disk and encode it as base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
def extract_sources(chunks: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta.get("source_file"), meta.get("page"))
        if key in seen:
            continue
        seen.add(key)
        image_path = meta.get("image_path") if meta.get("type") == "diagram" else None
        sources.append({
            "file": meta.get("source_file"),
            "page": meta.get("page"),
            "image_filename": os.path.basename(image_path) if image_path else None,
        })
    return sources
IMAGE_TOKEN_ESTIMATE = 1000
def fit_chunks_to_context(chunks: list[dict], max_context_tokens: int, reserved_for_answer: int = 1200) -> list[dict]:
    budget = max_context_tokens - reserved_for_answer
    kept = []
    running_total = 0

    for chunk in chunks:
        #changed:
        content_tokens = len(_tokenizer.encode(chunk["content"], add_special_tokens=False))
        has_image = chunk["metadata"].get("type") == "diagram" and chunk["metadata"].get("image_path")
        chunk_tokens = content_tokens + (IMAGE_TOKEN_ESTIMATE if has_image else 0)

        if running_total + chunk_tokens > budget and kept:
            break
        kept.append(chunk)
        running_total += chunk_tokens
    return kept

def retrieve_chunks(query: str, collection, embed_model, top_k: int = None) -> list[dict]:
    #Embed the query and pull back the most similar chunks from ChromaDB.
    top_k = top_k or config.TOP_K
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "embeddings"])

    chunks = []
    for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"content": content, "metadata": metadata})
    return chunks

def rerank_chunks(query: str, chunks: list[dict], top_n: int = 4) -> list[dict]:
    if len(chunks) <= top_n:
        return chunks
    
    pairs = [(query, c["content"]) for c in chunks]
    scores = _reranker.predict(pairs)
    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    file_counts = {}
    selected = []
    remaining = []
    for chunk, score in scored:
        file = chunk["metadata"].get("source_file")
        if file_counts.get(file, 0) >= 2:
            remaining.append((chunk, score))
            continue
        file_counts[file] = file_counts.get(file, 0) + 1
        selected.append((chunk, score))

    if len(selected) < top_n:
        selected += remaining[:top_n - len(selected)]
    selected.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in selected[:top_n]]
def build_prompt(query: str, chunks: list[dict], history_text: str = "", include_images: bool = True) -> tuple[str, list[str]]:
    #Assemble the full prompt: conversation history (if any) + retrieved context + question.
    text_context = ""
    images_b64 = []

    for chunk in chunks:
        meta = chunk["metadata"]
        text_context += f"--- Source ({meta['source_file']}, page {meta['page']}, type: {meta['type']}) ---\n"
        text_context += f"{chunk['content']}\n\n"

        if include_images and meta["type"] == "diagram" and meta.get("image_path"):
            images_b64.append(encode_image_base64(meta["image_path"]))

    history_section = f"CONVERSATION SO FAR:\n{history_text}\n\n" if history_text else ""
    prompt = f"""{history_section}Tu es un assistant scientifique rigoureux. Réponds à la question UNIQUEMENT d'après le contexte fourni ci-dessous.

RÈGLES:
1. Si l'information est présente (même sous une autre formulation ou dans un tableau), réponds de manière directe et concise.
2. Si le contexte ne contient AUCUN élément permettant de répondre, dis : "Les documents fournis ne contiennent pas cette information."
3. Si le contexte contient des éléments partiels ou indirects, utilise-les pour répondre au mieux — ne refuse pas par excès de prudence.
4. Copie les nombres, hyperparamètres et résultats EXACTEMENT comme écrits (n'arrondis pas, n'infère pas).
5. Si la question demande un chiffre, une dimension ou une valeur, cite-la explicitement avec son unité ou son contexte.
6. Réponds dans la même langue que la question.
7. Pas de tableaux Markdown (pas de |). Utilise des puces (-) et du texte brut.

CONTEXTE:
{text_context}

QUESTION: {query}

RÉPONSE:"""
    return prompt, images_b64

_TEMPLATE_TOKENS = len(build_prompt("", [], "", include_images=False)[0]) // 4
ANSWER_TOKEN_BUDGET = 500  # true free space intended for the model's generated answer text

def generate_answer(query: str, chunks: list[dict], backend: str = None, memory=None, groq_api_key: str = None) -> str :
    #Retrieve-then-generate: build the prompt (with history) and call the configured LLM backend.

    backend = (backend or config.LLM_BACKEND).lower()
    chunks = rerank_chunks(query, chunks, top_n=config.rerank_top_n)
    if backend == "groq":
        history_text = memory.get_history_text(max_turns=3) if memory else ""
        include_images = False  # Groq: rely on stored description only
    else:
        history_text = memory.get_history_text() if memory else ""
        include_images = True
        
    if backend == "ollama":
        history_tokens = len(history_text) // 3
        original_count = len(chunks)
        reserved = ANSWER_TOKEN_BUDGET + _TEMPLATE_TOKENS + history_tokens
        chunks = fit_chunks_to_context(chunks, config.CONTEXT_WINDOW, reserved_for_answer=reserved)
        if len(chunks) < original_count:
            logger.info(f"(Trimmed context: using {len(chunks)}/{original_count} retrieved chunks to fit Ollama's context window)")

    prompt, images_b64 = build_prompt(query, chunks, history_text, include_images=include_images)

    try:
        provider = llm_providers.get_llm_provider(backend, api_key=groq_api_key)
        answer = provider.generate(prompt, images=images_b64 if images_b64 else None)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"Sorry, I encountered an error while generating the answer: {e}"

    if memory is not None:
        memory.add_turn(query, answer)

    return answer
def generate_answer_stream(query: str, chunks: list[dict], backend: str = None, memory=None, sources_out: dict | None = None, groq_api_key: str = None):
    #Streaming responses: send each token to the browser as soon as it's generated.
    backend = (backend or config.LLM_BACKEND).lower()
    chunks = rerank_chunks(query, chunks, top_n=config.rerank_top_n)
    
    if backend == "groq":
        history_text = memory.get_history_text(max_turns=3) if memory else ""
        include_images = False
    else:
        history_text = memory.get_history_text() if memory else ""
        include_images = True

    if backend == "ollama":
        history_tokens = len(history_text) // 3
        original_count = len(chunks)
        reserved = ANSWER_TOKEN_BUDGET + _TEMPLATE_TOKENS + history_tokens
        chunks = fit_chunks_to_context(chunks, config.CONTEXT_WINDOW, reserved_for_answer=reserved)
        if len(chunks) < original_count:
            logger.info(f"Trimmed context: {len(chunks)}/{original_count}")
    if sources_out is not None:
        sources_out["sources"] = extract_sources(chunks)
    prompt, images_b64 = build_prompt(query, chunks, history_text, include_images=include_images)
    try:
        provider = llm_providers.get_llm_provider(backend, api_key=groq_api_key)
    except Exception as e:
        logger.error(f"LLM provider init failed: {e}")
        error_msg = f"\n[Error: {e}]"
        yield error_msg
        if memory is not None:
            memory.add_turn(query, error_msg)
        return

    full_answer = ""
    try:
        for token in provider.generate_stream(prompt, images=images_b64 if images_b64 else None):
            full_answer += token
            yield token
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        error_msg = f"\n[Error during generation: {e}]"
        full_answer += error_msg
        yield error_msg

    if memory is not None:
        memory.add_turn(query, full_answer)