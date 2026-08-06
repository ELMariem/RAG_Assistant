#Ingestion pipeline: turns a PDF into searchable chunks stored in ChromaDB.
import os
import io
import json
import time

import fitz  # PyMuPDF
from PIL import Image
from docling.document_converter import DocumentConverter
import logging
import config
import llm_providers

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

def is_significant_figure(image: Image.Image, ext: str) -> bool:
    #Decide whether an extracted image is a genuine diagram, or noise (equation fragments, decorative icons).
    width, height = image.size
    if ext == ".pdf":
        return width >= config.PDF_MIN_WIDTH_PX and height >= config.PDF_MIN_HEIGHT_PX
    if width < config.DOCX_MIN_WIDTH_PX or height < config.DOCX_MIN_HEIGHT_PX:
        return False
    ratio = max(width, height) / min(width, height)
    return ratio <= config.DOCX_MAX_ASPECT_RATIO


def get_page_no(item, default: int = 1) -> int:
    if item.prov and len(item.prov) > 0:
        return item.prov[0].page_no
    return default

def crop_figure(pdf_path: str, page_no: int, bbox, zoom: float = config.FIGURE_ZOOM) -> Image.Image:
    #Render a PDF page as an image and crop out just the figure region.
    pdf = fitz.open(pdf_path)
    page = pdf[page_no - 1]

    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    full_page_image = Image.open(io.BytesIO(pixmap.tobytes("png")))

    page_height = page.rect.height
    left = bbox.l * zoom
    right = bbox.r * zoom
    top = (page_height - bbox.t) * zoom
    bottom = (page_height - bbox.b) * zoom

    return full_page_image.crop((left, top, right, bottom))


def get_surrounding_text(doc, page_no: int, max_chars: int = config.CONTEXT_MAX_CHARS) -> str:

    page_texts = [t.text for t in doc.texts if get_page_no(t) == page_no]
    return " ".join(page_texts)[:max_chars]


def describe_diagram_with_context(image_path: str, context_text: str) -> str:
    #Ask the vision model to describe a diagram USING the surrounding text as context,
    
    prompt = f"""Here is text from the same page as this diagram:
\"\"\"{context_text}\"\"\"

Based on this context and the image, describe in detail what this diagram shows:
its components, labels, and connections, and how it relates to the surrounding text."""

    provider = llm_providers.get_llm_provider(config.LLM_BACKEND)
    return provider.generate(prompt, images=[image_path])


def row_to_sentence(row: dict) -> str:
    #Turn one table row (a dict) into a sentence.
    return "; ".join(f"{key}: {value}" for key, value in row.items())
    
def get_picture_image(doc, picture, source_path: str, ext: str):
    #Try to get the picture directly or Fall back to the render-and-crop trick only for PDFs.
    image = picture.get_image(doc)
    if image is not None:
        return image
    if ext == ".pdf":
        return crop_figure(source_path, picture.prov[0].page_no, picture.prov[0].bbox)
    return None

def process_document(file_path: str, figures_dir: str = None) -> list[dict]:
    #Route a file to the right processing function based on its extension.
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_EXTENSIONS:
        return process_docling_document(file_path, figures_dir)
    raise ValueError(f"Unsupported file type: {ext}")


def process_docling_document(file_path: str, figures_dir: str = None) -> list[dict]:
    #Parse one file and return a list of 'blocks' tagged by type.

    figures_dir = figures_dir or config.FIGURES_DIR  # falls back to shared default if not given
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"--- Processing {filename} ---")
    start_time = time.time()

    converter = DocumentConverter()
    doc = converter.convert(file_path).document
    blocks = []
    # Plain text blocks
    for item in doc.texts:
        blocks.append({
            "content": item.text,"type": "text",
            "page": get_page_no(item),"source_file": filename
        })
    #Tables: dual storage (sentences for embedding, JSON for precise answers)
    for table in doc.tables:
        records = table.export_to_dataframe(doc).to_dict(orient="records")
        sentence_version = " ".join(row_to_sentence(row) for row in records)

        blocks.append({
            "content": sentence_version,"type": "table",
            "page": get_page_no(table),"source_file": filename,
            "structured": json.dumps(records, ensure_ascii=False)
        })
    logger.info(f"  Text blocks: {len(doc.texts)} | Tables: {len(doc.tables)} | Diagrams: {len(doc.pictures)}")

    #Diagrams: crop, caption with context, keep image path for later
    os.makedirs(figures_dir, exist_ok=True)
    skipped_noise = 0
    skipped_unextractable = 0
    for i, picture in enumerate(doc.pictures):
        image = get_picture_image(doc, picture, file_path, ext)

        if image is None:
            skipped_unextractable += 1
            continue

        if not is_significant_figure(image, ext):
            skipped_noise += 1
            continue

        page_no = get_page_no(picture)
        logger.info(f"  Captioning diagram (page {page_no}, size {image.size})...")

        image_path = os.path.join(figures_dir, f"{filename}_page{page_no}_fig{i}.png")
        image.save(image_path)

        context = get_surrounding_text(doc, page_no)
        try:
            description = describe_diagram_with_context(image_path, context)
        except Exception as e:
            logger.warning(f"  Vision model failed for diagram on page {page_no}: {e}")
            description = f"[Diagram on page {page_no} — vision model unavailable]"

        blocks.append({
            "content": description, "type": "diagram",
            "page": page_no, "source_file": filename,
            "image_path": image_path
        })

    logger.info(
        f"  Kept {len(doc.pictures) - skipped_noise - skipped_unextractable} real diagrams "
        f"(skipped {skipped_noise} noise, {skipped_unextractable} unextractable) "
        f"out of {len(doc.pictures)} detected"
    )
    elapsed = time.time() - start_time
    logger.info(f"--- Finished {filename} in {elapsed:.1f}s, {len(blocks)} blocks ---")
    return blocks

def embed_and_store(blocks: list[dict], embed_model, client, collection_name: str = None) -> None:
    """Embed every block's content and store it in ChromaDB with its metadata."""
    collection_name = collection_name or config.COLLECTION_NAME
    collection = client.get_or_create_collection(collection_name)
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    for i, block in enumerate(blocks):
        embedding = embed_model.encode(block["content"]).tolist()
        metadata = {
            "type": block["type"],
            "page": block["page"],
            "source_file": block["source_file"]
        }
        if "structured" in block:
            metadata["structured"] = block["structured"]
        if "image_path" in block:
            metadata["image_path"] = block["image_path"]
        unique_id = f"{block['source_file']}_{i}_{block['type']}"
        ids.append(unique_id)
        embeddings.append(embedding)
        documents.append(block["content"])
        metadatas.append(metadata)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )