#FastAPI entry point — the HTTP API version of this assistant.

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import config
import rag
import memory as memory_module
from fastapi import UploadFile, File, Form
import os
import ingest

@asynccontextmanager
async def lifespan(app: FastAPI):
    #runs once, before the server accepts any requests
    print("Loading embedding model...")
    app.state.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    app.state.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    print("Ready to accept requests.")

    yield
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
class AskRequest(BaseModel):
    user_id: str
    question: str
    backend: str | None = None

class AskResponse(BaseModel):
    answer: str
    backend_used: str

def get_embed_model(request: Request):
    return request.app.state.embed_model

def get_chroma_client(request: Request):
    return request.app.state.chroma_client

class IngestResponse(BaseModel):
    filename: str
    blocks_created: int
    message: str

class ClearMemoryRequest(BaseModel):
    user_id: str

class ClearMemoryResponse(BaseModel):
    message: str
    new_conversation_id: int

def get_active_memory(user_id: str) -> memory_module.ConversationMemory:
    past_conversations = memory_module.list_conversations(user_id)
    if past_conversations:
        latest_id = past_conversations[0]["conversation_id"]
        return memory_module.ConversationMemory(user_id=user_id, conversation_id=latest_id)
    return memory_module.ConversationMemory(user_id=user_id)


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    embed_model=Depends(get_embed_model),
    client=Depends(get_chroma_client),
) -> AskResponse:
    collection_name = config.get_user_collection_name(body.user_id)
    collection = client.get_or_create_collection(collection_name)

    conversation_memory = get_active_memory(body.user_id)

    chunks = rag.retrieve_chunks(body.question, collection, embed_model)
    answer = rag.generate_answer(body.question, chunks, backend=body.backend, memory=conversation_memory)

    return AskResponse(
        answer=answer,
        backend_used=body.backend or config.LLM_BACKEND
    )
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    embed_model=Depends(get_embed_model),
    client=Depends(get_chroma_client),
) -> IngestResponse:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ingest.SUPPORTED_EXTENSIONS:
        return IngestResponse(
            filename=file.filename,
            blocks_created=0,
            message=f"Unsupported file type: {ext}"
        )

    data_dir = config.get_user_data_dir(user_id)
    figures_dir = config.get_user_figures_dir(user_id)
    collection_name = config.get_user_collection_name(user_id)
    os.makedirs(data_dir, exist_ok=True)

    dest_path = os.path.join(data_dir, file.filename)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    blocks = ingest.process_document(dest_path, figures_dir=figures_dir)
    ingest.embed_and_store(blocks, embed_model, client, collection_name=collection_name)

    return IngestResponse(
        filename=file.filename,
        blocks_created=len(blocks),
        message=f"Successfully ingested {len(blocks)} blocks."
    )
@app.post("/clear_memory", response_model=ClearMemoryResponse)
def clear_memory(body: ClearMemoryRequest) -> ClearMemoryResponse:
    conversation_memory = get_active_memory(body.user_id)
    conversation_memory.clear()

    return ClearMemoryResponse(
        message="Conversation history cleared. Starting fresh.",
        new_conversation_id=conversation_memory.conversation_id
    )