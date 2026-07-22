# all settings/constants in one place
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "documents")   # where your source PDFs live
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")  # cropped diagram images get saved here
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")         # ChromaDB's persistent storage folder

COLLECTION_NAME = "Alzeheimer"  # name of the ChromaDB collection (change this to switch datasets)

# Generator: answers the user's question using retrieved text + images
GENERATOR_MODEL = "qwen2.5vl:7b"

# Vision: describes diagrams/figures at ingestion time
VISION_MODEL = "qwen2.5vl:7b"

# Embedding: turns text into vectors for similarity search
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

FIGURE_ZOOM = 2.0          # render resolution
CONTEXT_MAX_CHARS = 800    # how much surrounding page text to feed the vision model for context

TOP_K = 4  # how many chunks ChromaDB returns per question

MAX_HISTORY_TURNS = 5  # how many past question/answer pairs to keep in the conversation buffer