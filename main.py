#FastAPI entry point — the HTTP API version of this assistant.

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import config
import rag
import memory as memory_module
import auth as auth_module
from fastapi import UploadFile, File, Form,Depends
import os
import ingest
import logging
import json
from datetime import datetime, timezone
import jwt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    #runs once, before the server accepts any requests
    logger.info("Loading embedding model...")
    app.state.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    app.state.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    logger.info("Ready to accept requests.")
    yield
    logger.info("Shutting down.")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

class AskRequest(BaseModel):
    user_id: str
    question: str
    backend: str | None = None
    conversation_id: int | None = None

class AskResponse(BaseModel):
    answer: str
    backend_used: str
    conversation_id: int

class IngestResponse(BaseModel):
    filename: str
    blocks_created: int
    message: str


class ClearMemoryResponse(BaseModel):
    message: str
    new_conversation_id: int

class ConversationSummary(BaseModel):
    conversation_id: int
    started_at: str
    preview: str

class MessageItem(BaseModel):
    role: str
    content: str

class DocumentItem(BaseModel):
    filename: str
    indexed: bool  # True if chunks exist in ChromaDB

class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]

class DeleteDocumentResponse(BaseModel):
    message: str
    filename: str

class RegisterRequest(BaseModel):
    user_id: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    user_id: str
    password: str

class NewConversationResponse(BaseModel):
    conversation_id: int
    message: str

class ClearConversationRequest(BaseModel):
    conversation_id: int


class ClearConversationResponse(BaseModel):
    message: str
    deleted_messages: int

def get_embed_model(request: Request):
    return request.app.state.embed_model

def get_chroma_client(request: Request):
    return request.app.state.chroma_client

def get_active_memory(user_id: str) -> memory_module.ConversationMemory:
    past_conversations = memory_module.list_conversations(user_id)
    if past_conversations:
        latest_id = past_conversations[0]["conversation_id"]
        return memory_module.ConversationMemory(user_id=user_id, conversation_id=latest_id)
    return memory_module.ConversationMemory(user_id=user_id)

def sanitize_filename(filename: str) -> str:
    #Remove path traversal characters from uploaded filenames.
    import re
    # Keep only safe characters: letters, numbers, dots, dashes, underscores
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', os.path.basename(filename))
    # Prevent hidden files and empty names
    safe = safe.lstrip('.')
    if not safe:
        safe = "uploaded_file"
    return safe

#Endpoints
@app.get("/")
def root():
    return RedirectResponse(url="/static/login.html")

@app.post("/auth/register")
def register(body: RegisterRequest):
    success = auth_module.register_user(body.user_id, body.password)
    if not success:
        raise HTTPException(status_code=409, detail="User already exists")
    return {"message": "User registered successfully"}


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    token = auth_module.authenticate_user(body.user_id, body.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return TokenResponse(access_token=token)

@app.get("/auth/me")
def auth_me(current_user: str = Depends(auth_module.get_current_user)):
    """
    Returns the currently authenticated user's ID.
    Use this to test if your token is working.
    """
    return {
        "user_id": current_user,
        "token_valid": True,
        "server_time_utc": datetime.now(timezone.utc).isoformat()
    }
"""
@app.post("/auth/debug-token")
def debug_token(token: str = Form(...)):

    try:
        # Decode without verification to inspect payload
        unverified = jwt.decode(token, options={"verify_signature": False})
        
        # Try full verification
        verified_user = auth_module.verify_token(token)
        
        return {
            "unverified_payload": unverified,
            "verified_user": verified_user,
            "secret_key_first_10": auth_module.SECRET_KEY[:10] + "...",
            "algorithm": auth_module.ALGORITHM,
            "token_valid": True
        }
    except Exception as e:
        return {
            "unverified_payload": jwt.decode(token, options={"verify_signature": False}) if token.count('.') == 2 else None,
            "error": str(e),
            "error_type": type(e).__name__,
            "token_valid": False
        }
"""
@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    current_user: str = Depends(auth_module.get_current_user),
    embed_model=Depends(get_embed_model),
    client=Depends(get_chroma_client),
) -> AskResponse:
    # ... use current_user instead of body.user_id for collection lookup ...
    collection_name = config.get_user_collection_name(current_user)
    collection = client.get_or_create_collection(collection_name)

    if body.conversation_id is not None:
        conversation_memory = memory_module.ConversationMemory(
            user_id=current_user, conversation_id=body.conversation_id
        )
    else:
        conversation_memory = get_active_memory(current_user)

    chunks = rag.retrieve_chunks(body.question, collection, embed_model)
    answer = rag.generate_answer(body.question, chunks, backend=body.backend, memory=conversation_memory)

    return AskResponse(
        answer=answer,
        backend_used=body.backend or config.LLM_BACKEND,
        conversation_id=conversation_memory.conversation_id
    )


@app.post("/ask_stream")
def ask_stream(
    body: AskRequest,
    current_user: str = Depends(auth_module.get_current_user),
    embed_model=Depends(get_embed_model),
    client=Depends(get_chroma_client),
):
    collection_name = config.get_user_collection_name(current_user)
    collection = client.get_or_create_collection(collection_name)

    if body.conversation_id is not None:
        conversation_memory = memory_module.ConversationMemory(
            user_id=current_user, conversation_id=body.conversation_id
        )
    else:
        conversation_memory = get_active_memory(current_user)

    chunks = rag.retrieve_chunks(body.question, collection, embed_model)
    
    def event_generator():
        for token in rag.generate_answer_stream(
            body.question, chunks, backend=body.backend, memory=conversation_memory
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_memory.conversation_id, 'backend_used': body.backend or config.LLM_BACKEND})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    current_user: str = Depends(auth_module.get_current_user),
    embed_model=Depends(get_embed_model),
    client=Depends(get_chroma_client),
) -> IngestResponse:
    """
    Upload and ingest a document. 
    The user is identified from the JWT token.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ingest.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {ingest.SUPPORTED_EXTENSIONS}"
        )
    
    safe_filename = sanitize_filename(file.filename)
    data_dir = config.get_user_data_dir(current_user)
    figures_dir = config.get_user_figures_dir(current_user)
    collection_name = config.get_user_collection_name(current_user)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    dest_path = os.path.join(data_dir, safe_filename)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    try:
        blocks = ingest.process_document(dest_path, figures_dir=figures_dir)
        ingest.embed_and_store(blocks, embed_model, client, collection_name=collection_name)
    except Exception as e:
        logger.error(f"Ingestion failed for {safe_filename}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResponse(
        filename=safe_filename,
        blocks_created=len(blocks),
        message=f"Successfully ingested {len(blocks)} blocks."
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(
    current_user: str = Depends(auth_module.get_current_user),
    client=Depends(get_chroma_client),
):
    """List all documents for the authenticated user."""
    data_dir = config.get_user_data_dir(current_user)
    collection_name = config.get_user_collection_name(current_user)
    collection = client.get_or_create_collection(collection_name)
    
    if os.path.exists(data_dir):
        disk_files = {
            f for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in ingest.SUPPORTED_EXTENSIONS
        }
    else:
        disk_files = set()
    
    try:
        existing = collection.get(include=["metadatas"])
        indexed_files = {
            meta["source_file"] for meta in existing["metadatas"]
            if meta.get("source_file")
        }
    except Exception as e:
        logger.warning(f"Could not fetch indexed files: {e}")
        indexed_files = set()
    
    all_files = disk_files | indexed_files
    documents = [
        DocumentItem(filename=f, indexed=f in indexed_files)
        for f in sorted(all_files)
    ]
    
    return DocumentListResponse(documents=documents)


@app.delete("/documents/{filename}", response_model=DeleteDocumentResponse)
def delete_document(
    filename: str,
    current_user: str = Depends(auth_module.get_current_user),
    client=Depends(get_chroma_client),
):
    """Delete a document: remove from disk, ChromaDB, and figures folder."""
    safe_filename = sanitize_filename(filename)
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    data_dir = config.get_user_data_dir(current_user)
    figures_dir = config.get_user_figures_dir(current_user)
    collection_name = config.get_user_collection_name(current_user)
    collection = client.get_or_create_collection(collection_name)
    
    file_path = os.path.join(data_dir, safe_filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted file from disk: {file_path}")
    
    try:
        collection.delete(where={"source_file": safe_filename})
        logger.info(f"Deleted chunks from ChromaDB for: {safe_filename}")
    except Exception as e:
        logger.error(f"Failed to delete from ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete from database: {e}")
    
    if os.path.exists(figures_dir):
        deleted_figs = 0
        for fig in os.listdir(figures_dir):
            if fig.startswith(safe_filename + "_"):
                os.remove(os.path.join(figures_dir, fig))
                deleted_figs += 1
        if deleted_figs:
            logger.info(f"Deleted {deleted_figs} figures for: {safe_filename}")
    
    return DeleteDocumentResponse(
        message=f"Document '{safe_filename}' deleted successfully.",
        filename=safe_filename
    )


@app.post("/clear_memory", response_model=ClearMemoryResponse)
def clear_memory(
    current_user: str = Depends(auth_module.get_current_user)
) -> ClearMemoryResponse:
    conversation_memory = get_active_memory(current_user)
    conversation_memory.clear()
    return ClearMemoryResponse(
        message="Conversation history cleared. Starting fresh.",
        new_conversation_id=conversation_memory.conversation_id
    )

@app.get("/conversations", response_model=list[ConversationSummary])
def get_conversations(
    current_user: str = Depends(auth_module.get_current_user),
) -> list[ConversationSummary]:
    conversations = memory_module.list_conversations(current_user)
    return [
        ConversationSummary(
            conversation_id=c["conversation_id"],
            started_at=c["started_at"].isoformat()if c["started_at"] else "",
            preview=memory_module.get_conversation_preview(c["conversation_id"])[:60]
        )
        for c in conversations
    ]

@app.post("/new_conversation", response_model=NewConversationResponse)
def new_conversation(
    current_user: str = Depends(auth_module.get_current_user),
) -> NewConversationResponse:
    new_id = memory_module.create_conversation(current_user)
    return NewConversationResponse(
        conversation_id=new_id,
        message="New conversation started."
    )

@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageItem])
def get_conversation_messages(
    conversation_id: int,
    current_user: str = Depends(auth_module.get_current_user),
) -> list[MessageItem]:
    # Optional: verify this conversation belongs to current_user
    with memory_module.engine.connect() as conn:
            result = conn.execute(
                memory_module.text("""
                    SELECT user_id FROM conversations
                    WHERE conversation_id = :conversation_id
                """),
                {"conversation_id": conversation_id}
            )
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if row[0] != current_user:
                raise HTTPException(status_code=403, detail="Not your conversation")
    messages = memory_module.get_all_messages(conversation_id)
    return [MessageItem(**m) for m in messages]

@app.post("/conversations/{conversation_id}/clear", response_model=ClearConversationResponse)
def clear_conversation(
    conversation_id: int,
    current_user: str = Depends(auth_module.get_current_user),
) -> ClearConversationResponse:
    #Supprime tous les messages d'une conversation spécifique.
    with memory_module.engine.connect() as conn:
        result = conn.execute(
            memory_module.text("""
                SELECT user_id FROM conversations
                WHERE conversation_id = :conversation_id
            """),
            {"conversation_id": conversation_id}
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if row[0] != current_user:
            raise HTTPException(status_code=403, detail="Not your conversation")
    with memory_module.engine.begin() as conn:
        count_result = conn.execute(
            memory_module.text("""
                SELECT COUNT(*) FROM messages
                WHERE conversation_id = :conversation_id
            """),
            {"conversation_id": conversation_id}
        )
        deleted_count = count_result.scalar()
        conn.execute(
            memory_module.text("""
                DELETE FROM messages
                WHERE conversation_id = :conversation_id
            """),
            {"conversation_id": conversation_id}
        )
        conn.execute(
            memory_module.text("""
                DELETE FROM conversations 
                WHERE conversation_id = :conversation_id
            """),
            {"conversation_id": conversation_id}
        )
    return ClearConversationResponse(
        message=f"Conversation {conversation_id} cleared.",
        deleted_messages=deleted_count
    )