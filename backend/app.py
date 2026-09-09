#Entry point: ingests any new PDFs found in data/documents, then runs an interactive question loop.

import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
import shutil
import config
import ingest
import rag
import memory as memory_module

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_already_ingested_files(collection) -> set:
    existing = collection.get(include=["metadatas"])
    return {meta["source_file"] for meta in existing["metadatas"] if meta.get("source_file")}


def sync_collection(embed_model, client, user_id: str):
    #Make sure every PDF currently in data/documents is represented in ChromaDB.
    
    data_dir = config.get_user_data_dir(user_id)
    figures_dir = config.get_user_figures_dir(user_id)
    collection_name = config.get_user_collection_name(user_id)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    collection = client.get_or_create_collection(collection_name)
    existing = collection.get(include=["metadatas"])
    already_ingested = {m["source_file"] for m in existing["metadatas"] if m.get("source_file")}

    print(f"Files in folder: {os.listdir(data_dir)}")
    print(f"Already ingested: {already_ingested}")

    all_files = [
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in ingest.SUPPORTED_EXTENSIONS]
    new_files = [f for f in all_files if f not in already_ingested]

    if not new_files:
        print(f"Collection up to date ({len(already_ingested)} file(s) already ingested).")
        return collection

    print(f"Found {len(new_files)} new file(s) to ingest: {new_files}")
    for filename in new_files:
        file_path = os.path.join(data_dir, filename)
        blocks = ingest.process_document(file_path, figures_dir=figures_dir)
        ingest.embed_and_store(blocks, embed_model, client,  collection_name=collection_name)
    return collection


def main():
    print("Loading models (embedding model + connecting to ChromaDB)...")
    memory_module.init_db()
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    user_id = input("Enter your username: ").strip() or "default_user"

    start = time.time()
    collection = sync_collection(embed_model,  client, user_id)
    print(f"Ready in {time.time() - start:.1f}s.\n")

    current_backend = config.LLM_BACKEND
    past_conversations = memory_module.list_conversations(user_id)

    if past_conversations:
        conversation_memory = memory_module.ConversationMemory(user_id=user_id, conversation_id=past_conversations[0]["conversation_id"])
        print(f"Welcome back, {user_id}. Resuming your most recent conversation.\n")
    else:
        conversation_memory = memory_module.ConversationMemory(user_id=user_id)
        print(f"Welcome, {user_id}. Starting a new conversation.\n")

    print("Assistant ready. Type 'exit' to quit, 'add + path' to add documents, ""'new conversation' to start fresh, 'clear memory' to reset current session.\n")
    while True:
        query = input("You: ")
        if query.lower() in ("exit", "quit"):
            break
        if query.lower() == "new conversation":
            conversation_memory.clear()
            print("Started a new conversation.\n")
            continue
        if query.lower() == "clear memory":
            conversation_memory.clear()
            print("Conversation history cleared.\n")
            continue
        if query.lower().startswith("add "):
            source_path = query[4:].strip().strip('"')
            if not os.path.isfile(source_path):
                print(f"File not found: {source_path}\n")
                continue
            dest_path = os.path.join(config.get_user_data_dir(user_id), os.path.basename(source_path))
            shutil.copy(source_path, dest_path)
            print("Copied. Checking for new files...")
            collection = sync_collection(embed_model, client, user_id)
            continue
        
        if query.lower() in ("use ollama", "use groq"):
            current_backend = query.lower().split()[-1]
            print(f"Switched to {current_backend} backend.\n")
            continue

        chunks = rag.retrieve_chunks(query, collection, embed_model)
        answer = rag.generate_answer(query, chunks, backend=current_backend,  memory=conversation_memory)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()