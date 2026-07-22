"""
RAG chain: given a user question, retrieve relevant chunks from ChromaDB,
then ask the generator model to answer using that context (and any attached images).
"""

import base64
import ollama
import config


def encode_image_base64(path: str) -> str:
    """Read an image file from disk and encode it as base64, ready to send to Ollama."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def retrieve_chunks(query: str, collection, embed_model, top_k: int = None) -> list[dict]:
    """Embed the query and pull back the most similar chunks from ChromaDB."""
    top_k = top_k or config.TOP_K
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"content": content, "metadata": metadata})
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Assemble the text context from all retrieved chunks, and collect base64 images
    for any chunk that is a diagram (so the model can look at it directly, not just
    rely on its stored description).
    """
    text_context = ""
    images_b64 = []

    for chunk in chunks:
        meta = chunk["metadata"]
        text_context += f"--- Source ({meta['source_file']}, page {meta['page']}, type: {meta['type']}) ---\n"
        text_context += f"{chunk['content']}\n\n"

        if meta["type"] == "diagram" and meta.get("image_path"):
            images_b64.append(encode_image_base64(meta["image_path"]))

    prompt = f"""Based on the following context, answer the question clearly.

CONTEXT:
{text_context}

QUESTION: {query}

If an image is provided alongside the text, look at it directly to verify or add precision —
don't rely only on the text description of it. If the context is insufficient, say so.

ANSWER:"""

    return prompt, images_b64


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Retrieve-then-generate: build the prompt and call the generator model."""
    prompt, images_b64 = build_prompt(query, chunks)

    message = {"role": "user", "content": prompt}
    if images_b64:
        message["images"] = images_b64  # only attached when a diagram was actually retrieved

    response = ollama.chat(model=config.GENERATOR_MODEL, messages=[message])
    return response["message"]["content"]