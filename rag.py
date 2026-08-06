"""
RAG chain: given a user question, retrieve relevant chunks from ChromaDB,
then ask the generator model to answer using that context (and any attached images)."""

import base64
import config
import llm_providers
import logging

logger = logging.getLogger(__name__)

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
        content_tokens = len(chunk["content"]) // 4
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

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"content": content, "metadata": metadata})
    return chunks


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

If the question refers back to something discussed earlier in the conversation
(like "it", "that model", "the same dataset"), use the conversation history above
to understand what's being referred to.
If an image is provided alongside the text, look at it directly to verify or add precision —
don't rely only on the text description of it. If the context is insufficient, say so.

ANSWER:"""

    return prompt, images_b64


def generate_answer(query: str, chunks: list[dict], backend: str = None, memory=None) -> str:
    #Retrieve-then-generate: build the prompt (with history) and call the configured LLM backend.

    backend = (backend or config.LLM_BACKEND).lower()

    if backend == "groq":
        history_text = memory.get_history_text(max_turns=3) if memory else ""
        include_images = False  # Groq: rely on stored description only
    else:
        history_text = memory.get_history_text() if memory else ""
        include_images = True
        
    if backend == "ollama":
        history_tokens = len(history_text) // 4
        original_count = len(chunks)
        chunks = fit_chunks_to_context(chunks, config.CONTEXT_WINDOW, reserved_for_answer=1200 + history_tokens)
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
    if backend == "groq":
        history_text = memory.get_history_text(max_turns=3) if memory else ""
        include_images = False
    else:
        history_text = memory.get_history_text() if memory else ""
        include_images = True

    if backend == "ollama":
        history_tokens = len(history_text) // 4
        original_count = len(chunks)
        chunks = fit_chunks_to_context(chunks, config.CONTEXT_WINDOW, reserved_for_answer=1200 + history_tokens)
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