# all settings/constants in one place
import os

from dotenv import load_dotenv
load_dotenv()
print("Groq key loaded:", "GROQ_API_KEY" in os.environ)  # should print True if the key is loaded
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#print(">python run_evaluation.py --user_id elabed --benchmark benchmark2.json --backend ollama --judge groq")
#python run_evaluation.py --user_id admin --benchmark Benchmark2.json --backend ollama --judge_backend groq --judge_model "openai/gpt-oss-120b"
DATA_DIR = os.path.join(BASE_DIR, "data", "documents")   # where your source PDFs live
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")  # cropped diagram images get saved here
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")         # ChromaDB's persistent storage folder

COLLECTION_NAME = "Alzeheimer"  # name of the ChromaDB collection
CONTEXT_WINDOW = 8192
# Generator: answers the user's question using retrieved text + images
GENERATOR_MODEL = "qwen2.5vl:7b"

# Vision: describes diagrams/figures at ingestion time
VISION_MODEL = "qwen2.5vl:7b"
# Embedding: turns text into vectors for similarity search
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

FIGURE_ZOOM = 3.0          # render resolution
CONTEXT_MAX_CHARS = 800    # how much surrounding page text to feed the vision model for context

TOP_K =40 # how many chunks ChromaDB returns per question
rerank_top_n=10

MAX_HISTORY_TURNS = 5  # how many past question/answer pairs to keep in the conversation buffer
#filter out tiny figures
PDF_MIN_WIDTH_PX = 80
PDF_MIN_HEIGHT_PX = 80

DOCX_MIN_WIDTH_PX = 200
DOCX_MIN_HEIGHT_PX = 200
DOCX_MAX_ASPECT_RATIO = 1.7

# LLM BACKEND
LLM_BACKEND = "ollama"   # "ollama" (local/private) or "groq" (cloud/fast)
GROQ_MODEL = "openai/gpt-oss-120b"        # model used for previous tests
#GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"     # image-attached questions — separate multimodal model

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 1
#database configuration
DB_PATH = os.path.join(BASE_DIR, "memory.db")

CHUNK_TOKEN_SIZE = 110      # the embedding model's 128-token max_seq_length
CHUNK_TOKEN_OVERLAP = 20
MAX_TABLE_ROWS_PER_CHUNK = 15

def get_user_data_dir(user_id: str) -> str:
    return os.path.join(DATA_DIR, user_id)

def get_user_figures_dir(user_id: str) -> str:
    return os.path.join(FIGURES_DIR, user_id)

def get_user_collection_name(user_id: str) -> str:
    return f"{COLLECTION_NAME}_{user_id}"