"""
Ingestion pipeline: turns a PDF into searchable chunks stored in ChromaDB.

Steps for each PDF:
1. Parse it with Docling -> get text blocks, tables, and picture locations.
2. Plain text: used as-is.
3. Tables: converted to sentences (for embedding) + raw JSON kept (for precise answers later).
4. Pictures: cropped from the page, described by a vision model using surrounding text as
context, and that description becomes the chunk's content.
5. Every chunk is embedded and stored in ChromaDB with metadata (type, page, source file,
and anything needed to reconstruct precise context later).
"""

import os
import io
import json
import time

import fitz  # PyMuPDF
import ollama
from PIL import Image
from docling.document_converter import DocumentConverter

import config
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

def is_significant_figure(image, min_width=200, min_height=200, max_aspect_ratio=1.7) -> bool:
    """
    Keep only real diagrams/charts, filtering out:
    - tiny equation symbols (too small — min_width/min_height catch these)
    - multi-symbol equation lines (too elongated — aspect ratio catches these,
    since a formula renders as one long thin strip, not a square-ish figure)
    """
    width, height = image.size
    if width < min_width or height < min_height:
        return False

    longer_side = max(width, height)
    shorter_side = min(width, height)
    aspect_ratio = longer_side / shorter_side

    return aspect_ratio <= max_aspect_ratio


def get_page_no(item, default: int = 1) -> int:
    """
    Safely get the page number of a Docling item.
    PDFs always have provenance info; DOCX elements sometimes don't
    """
    if item.prov and len(item.prov) > 0:
        return item.prov[0].page_no
    return default
def crop_figure(pdf_path: str, page_no: int, bbox, zoom: float = config.FIGURE_ZOOM) -> Image.Image:
    """
    Render a PDF page as an image and crop out just the figure region.
    Needed because Docling detects WHERE a figure is, but many diagrams are vector
    drawings, not embedded image files, so there's nothing to extract directly.
    """
    pdf = fitz.open(pdf_path)
    page = pdf[page_no - 1]  # Docling pages are 1-indexed, PyMuPDF pages are 0-indexed

    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    full_page_image = Image.open(io.BytesIO(pixmap.tobytes("png")))

    # Docling's bbox uses PDF coordinates (origin at bottom-left).
    # Images use top-left origin, so we flip the y-axis here.
    page_height = page.rect.height
    left = bbox.l * zoom
    right = bbox.r * zoom
    top = (page_height - bbox.t) * zoom
    bottom = (page_height - bbox.b) * zoom

    return full_page_image.crop((left, top, right, bottom))


def get_surrounding_text(doc, page_no: int, max_chars: int = config.CONTEXT_MAX_CHARS) -> str:
    """
    Collect the plain text on the same page as a figure, so the vision model can
    connect labels in a diagram to concepts explained nearby, instead of describing it blind.
    """
    page_texts = [t.text for t in doc.texts if get_page_no(t) == page_no]
    return " ".join(page_texts)[:max_chars]


def describe_diagram_with_context(image_path: str, context_text: str) -> str:
    """
    Ask the vision model to describe a diagram USING the surrounding text as context,
    and to suggest questions it could answer (helps retrieval match real questions later).
    """
    prompt = f"""Here is text from the same page as this diagram:
\"\"\"{context_text}\"\"\"

Based on this context and the image, describe in detail what this diagram shows:
its components, labels, and connections, and how it relates to the surrounding text."""

    response = ollama.chat(model=config.VISION_MODEL, messages=[{
        "role": "user",
        "content": prompt,
        "images": [image_path]
    }])
    return response["message"]["content"]


def row_to_sentence(row: dict) -> str:
    """Turn one table row (a dict) into a plain sentence, better suited for embedding than raw JSON."""
    return "; ".join(f"{key}: {value}" for key, value in row.items())

def process_document(file_path: str) -> list[dict]:
    """Route a file to the right processing function based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".pdf", ".docx"):
        return process_docling_document(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
def get_picture_image(doc, picture, source_path: str, ext: str):
    """
    Try to get the picture directly (works for DOCX, where images are usually embedded).
    Fall back to the render-and-crop trick only for PDFs (needed for vector-drawn diagrams).
    """
    image = picture.get_image(doc)
    if image is not None:
        return image
    if ext == ".pdf":
        return crop_figure(source_path, picture.prov[0].page_no, picture.prov[0].bbox)
    return None  # couldn't extract — will be skipped with a warning

def process_docling_document(file_path: str) -> list[dict]:
    """
    Parse one file and return a list of 'blocks' — a unified representation where
    text, tables, and diagrams all end up as plain text content, tagged by type.
    Every block also carries the source filename, used later to avoid re-ingesting
    a file that's already in the database.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    print(f"\n--- Processing {filename} ---")
    start_time = time.time()

    converter = DocumentConverter()
    doc = converter.convert(file_path).document
    blocks = []

    # --- Plain text blocks ---
    for item in doc.texts:
        blocks.append({
            "content": item.text,"type": "text",
            "page": get_page_no(item),"source_file": filename
        })

    # --- Tables: dual storage (sentences for embedding, JSON for precise answers) ---
    for table in doc.tables:
        records = table.export_to_dataframe().to_dict(orient="records")
        sentence_version = " ".join(row_to_sentence(row) for row in records)

        blocks.append({
            "content": sentence_version,"type": "table",
            "page": get_page_no(table),"source_file": filename,
            "structured": json.dumps(records, ensure_ascii=False)  # kept for the LLM at answer time
        })

    print(f"  Text blocks: {len(doc.texts)} | Tables: {len(doc.tables)} | Diagrams: {len(doc.pictures)}")

    # --- Diagrams: crop, caption with context, keep image path for later ---
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    skipped_small = 0
    for i, picture in enumerate(doc.pictures):
        bbox = picture.prov[0].bbox if picture.prov else None
        if bbox is None or not is_significant_figure(bbox):
            skipped_small += 1
            continue

        page_no = get_page_no(picture)
        print(f"  Captioning diagram {i + 1}/{len(doc.pictures)} (page {page_no})...")

        image = get_picture_image(doc, picture, file_path, ext)
        if image is None:
            print(f"    Skipped — could not extract this image.")
            continue
        
        image_path = os.path.join(config.FIGURES_DIR, f"{filename}_page{page_no}_fig{i}.png")
        image.save(image_path)

        context = get_surrounding_text(doc, page_no)
        description = describe_diagram_with_context(image_path, context)

        blocks.append({
            "content": description,"type": "diagram",
            "page": page_no,"source_file": filename,
            "image_path": image_path  # used at answer time to re-attach the real image
        })
    print(f"  Skipped {skipped_small} small/decorative figures out of {len(doc.pictures)} detected")
    elapsed = time.time() - start_time
    print(f"--- Finished {filename} in {elapsed:.1f}s, {len(blocks)} blocks ---")
    return blocks


def embed_and_store(blocks: list[dict], embed_model, client) -> None:
    """Embed every block's content and store it in ChromaDB with its metadata."""
    collection = client.get_or_create_collection(config.COLLECTION_NAME)

    for i, block in enumerate(blocks):
        embedding = embed_model.encode(block["content"]).tolist()

        # Only these simple types (str/int/float/bool) are allowed as ChromaDB metadata values
        metadata = {
            "type": block["type"],
            "page": block["page"],
            "source_file": block["source_file"]
        }
        if "structured" in block:
            metadata["structured"] = block["structured"]
        if "image_path" in block:
            metadata["image_path"] = block["image_path"]

        # Unique id combining filename so two files can't overwrite each other's chunks
        unique_id = f"{block['source_file']}_{i}_{block['type']}"

        collection.add(
            ids=[unique_id],
            embeddings=[embedding],
            documents=[block["content"]],
            metadatas=[metadata]
        )