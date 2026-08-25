"""
RAG chain: given a user question, retrieve relevant chunks from ChromaDB,
then ask the generator model to answer using that context (and any attached images)."""

import base64
import config
import llm_providers
import logging
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)
_reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
def encode_image_base64(path: str) -> str:
    """Read an image file from disk and encode it as base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
#trim dynamically based on real token estimate, not a fixed number (max = top_k=4)
IMAGE_TOKEN_ESTIMATE = 1000
def fit_chunks_to_context(chunks: list[dict], max_context_tokens: int, reserved_for_answer: int = 1200) -> list[dict]:
    budget = max_context_tokens - reserved_for_answer
    kept = []
    running_total = 0

    for chunk in chunks:
        content_tokens = len(chunk["content"]) // 3
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
    seen_keys = set()
    best_per_key = []
    remaining = []
    for chunk, score in scored:
        meta = chunk["metadata"]
        key = (meta.get("source_file"), meta.get("page"), meta.get("type"))
        if key not in seen_keys:
            seen_keys.add(key)
            best_per_key.append((chunk, score))
        else:
            remaining.append((chunk, score))

    selected = best_per_key[:top_n]
    if len(selected) < top_n:
        selected += remaining[:top_n - len(selected)]

    # Re-sort by score so the strongest evidence still comes first in the prompt --
    # diversification only changes WHICH chunks are kept, not their presentation order.
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
    prompt = f"""{history_section}Based on the following context, answer the question clearly.
CONTEXT:
{text_context}
 
QUESTION: {query}
 
PRECISION:
- Copy exact numbers (scores, hyperparameters, counts, measurements) exactly as written — never round, approximate, or infer a figure that isn't visibly present.
 
WHEN THE CONTEXT IS INSUFFICIENT (read carefully):
- Judge sufficiency on SUBSTANCE, not wording: if the answer is present in different words, in a table row, or across two adjacent sentences, that counts — don't refuse just because it isn't phrased like the question. Loosely related content is NOT the same as an answer, though.
- If the substance is genuinely absent, say so plainly ("the documents don't provide this information") instead of guessing or restating vague context as an answer — an honest gap always beats a plausible-sounding invention.
 
FOCUS (applies only once you've confirmed the context above actually answers the question):
- Answer only what's asked — no unrequested background, related facts, or interpretation, even if true and present in the context.
- Lead with the direct answer in the first sentence; don't preface with setup or restated context.
- This governs HOW MUCH to say once you have an answer, not whether one exists — that judgment call belongs entirely to the section above.
 
LANGUAGE:
- Answer in the SAME LANGUAGE as the question above. If the question is in French, answer in French; if in English, answer in English. Do not translate or switch languages.
 
CONVERSATION CONTEXT:
- If the question refers back to something discussed earlier ("it", "that model", "the same dataset"), use the conversation history above to resolve what's being referred to.
 
IMAGES:
- If an image is provided alongside the text, look at it directly to verify or add precision — don't rely only on the text description of it.
 
FORMATTING (this answer will be displayed in a plain chat bubble):
- No LaTeX/math notation (\\( \\), \\frac{{}}) — write formulas in plain text, e.g. "Recall = TP / (TP + FN)". No Markdown tables (no | pipes); use a short bullet list instead.
- Use short paragraphs and bullet points ("-"), not one dense block of text. Bold only key terms, not full sentences.
 
ANSWER:"""
    return prompt, images_b64

_TEMPLATE_TOKENS = len(build_prompt("", [], "", include_images=False)[0]) // 4
ANSWER_TOKEN_BUDGET = 500  # true free space intended for the model's generated answer text

def generate_answer(query: str, chunks: list[dict], backend: str = None, memory=None) -> str:
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
        provider = llm_providers.get_llm_provider(backend)
        answer = provider.generate(prompt, images=images_b64 if images_b64 else None)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"Sorry, I encountered an error while generating the answer: {e}"

    if memory is not None:
        memory.add_turn(query, answer)

    return answer
def generate_answer_stream(query: str, chunks: list[dict], backend: str = None, memory=None):
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

    prompt, images_b64 = build_prompt(query, chunks, history_text, include_images=include_images)
    provider = llm_providers.get_llm_provider(backend)

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