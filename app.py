"""
Entry point: ingests any new PDFs found in data/documents (skipping ones already
processed), then runs an interactive question loop.
Memory will be added here once memory.py is built.
"""

import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
import shutil
import config
import ingest
import rag


def get_already_ingested_files(collection) -> set:
    """Look at what's already stored in ChromaDB and return the set of source filenames."""
    existing = collection.get(include=["metadatas"])
    return {meta["source_file"] for meta in existing["metadatas"] if meta.get("source_file")}


def sync_collection(embed_model, client):
    """
    Make sure every PDF currently in data/documents is represented in ChromaDB.
    Already-ingested files are skipped; new files are processed and added.
    This is what lets you drop in a new PDF later without wiping the database.
    """
    collection = client.get_or_create_collection(config.COLLECTION_NAME)
    already_ingested = get_already_ingested_files(collection)

    print(f"Files in folder: {os.listdir(config.DATA_DIR)}")
    print(f"Already ingested: {already_ingested}")

    all_files = [
        f for f in os.listdir(config.DATA_DIR)
        if os.path.splitext(f)[1].lower() in ingest.SUPPORTED_EXTENSIONS]
    new_files = [f for f in all_files if f not in already_ingested]

    if not new_files:
        print(f"Collection up to date ({len(already_ingested)} file(s) already ingested).")
        return collection

    print(f"Found {len(new_files)} new file(s) to ingest: {new_files}")
    for filename in new_files:
        file_path = os.path.join(config.DATA_DIR, filename)
        blocks = ingest.process_document(file_path)
        ingest.embed_and_store(blocks, embed_model, client)
    return collection


def main():
    print("Loading models (embedding model + connecting to ChromaDB)...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    start = time.time()
    collection = sync_collection(embed_model, client)
    print(f"Ready in {time.time() - start:.1f}s.\n")

    print("Assistant ready. Type 'exit' to quit, 'add + path' to add new documents.\n")
    while True:
        query = input("You: ")
        if query.lower() in ("exit", "quit"):
            break
        if query.lower().startswith("add "):
            source_path = query[4:].strip().strip('"')  # allows pasting a path with quotes
            if not os.path.isfile(source_path):
                print(f"File not found: {source_path}\n")
                continue
            dest_path = os.path.join(config.DATA_DIR, os.path.basename(source_path))
            shutil.copy(source_path, dest_path)
        
            print(f"Copied to data/documents. Checking for new files...")
            collection = sync_collection(embed_model, client)
            continue
        chunks = rag.retrieve_chunks(query, collection, embed_model)
        answer = rag.generate_answer(query, chunks)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()