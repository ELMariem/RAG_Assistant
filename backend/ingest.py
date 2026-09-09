#Ingestion pipeline: turns a PDF into searchable chunks stored in ChromaDB.
import os
import io
import json
import time
import re
from transformers import AutoTokenizer
import fitz  # PyMuPDF
from PIL import Image
from docling.document_converter import DocumentConverter
import logging
import config
import llm_providers
from collections import defaultdict

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
HEADING_NUM_RE = re.compile(r'^(\d+(?:\.\d+)*)[\.\)]?\s+(.*)')

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

def _looks_like_prompt_echo(description: str) -> bool:
    """
    Detect when the vision model echoed our own instruction back instead of actually
    describing the image
    """
    lowered = description.lower()
    markers = [
        "retranscris verbatim", "sans paraphraser",
        "ne développe pas au-delà", "densité d'information",
    ]
    return any(marker in lowered for marker in markers)

def describe_diagram_with_context(image_path: str, context_text: str) -> str:
    prompt = f"""Voici le texte environnant sur la même page que ce diagramme :
\"\"\"{context_text}\"\"\"
Décris ce diagramme en respectant cette structure, de façon CONCISE (maximum ~150 mots au total) :

1. TEXTE VISIBLE : liste, sans phrases, tout texte/label/légende/chiffre lisible dans l'image
   (juste les éléments bruts, pas de description autour). Cette partie est la plus importante.
2. STRUCTURE : une ou deux phrases maximum sur les composants et leurs connexions.
3. LIEN AVEC LE CONTEXTE : une phrase maximum.
Ne développe pas au-delà de ce qui est nécessaire — privilégie la densité d'information sur la
longueur du texte.
IMPORTANT : rédige ta réponse dans la MÊME LANGUE que le texte de contexte ci-dessus."""
    provider = llm_providers.get_llm_provider(config.LLM_BACKEND)
    return provider.generate(prompt, images=[image_path])

def get_picture_caption(doc, picture, page_no: int) -> str:
    caption = picture.caption_text(doc) if hasattr(picture, "caption_text") else ""
    if caption:
        return caption.strip()
    for item, _ in doc.iterate_items():
        if "CAPTION" in str(getattr(item, "label", "")) and get_page_no(item) == page_no:
            return (getattr(item, "text", "") or "").strip()
    return ""


def process_diagrams(doc, file_path: str, filename: str, figures_dir: str) -> list[dict]:

    ext = os.path.splitext(file_path)[1].lower()
    os.makedirs(figures_dir, exist_ok=True)
    blocks = []
    skipped_noise = skipped_unextractable = 0
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
            if _looks_like_prompt_echo(description):
                logger.warning(f"  Vision model echoed the instruction instead of describing "
                                f"diagram on page {page_no} -- retrying once")
                description = describe_diagram_with_context(image_path, context)
                if _looks_like_prompt_echo(description):
                    caption = get_picture_caption(doc, picture, page_no)
                    logger.warning(f"  Retry also echoed on page {page_no} -- falling back to caption "
                                    f"({'found' if caption else 'NONE found'}); check the crop image, "
                                    f"likely a source-image problem, not a prompt problem")
                    description = caption or f"Figure, page {page_no} de {filename}."
            MAX_DIAGRAM_DESCRIPTION_CHARS = 1400
            if len(description) > MAX_DIAGRAM_DESCRIPTION_CHARS:
                logger.warning(f"  Diagram description on page {page_no} was {len(description)} chars "
                                f"-- truncated to {MAX_DIAGRAM_DESCRIPTION_CHARS}")
                description = description[:MAX_DIAGRAM_DESCRIPTION_CHARS].rstrip() + " [...]"
        except Exception as e:
            caption = get_picture_caption(doc, picture, page_no)
            logger.warning(f"  Vision model failed for diagram on page {page_no}: {e} "
                            f"-- falling back to caption ({'found' if caption else 'NONE found'})")
            description = caption or f"Figure, page {page_no} de {filename}."
        blocks.append({
            "content": description, "type": "diagram", "page": page_no,
            "source_file": filename, "image_path": image_path,
        })
    logger.info(f"  Diagrams: {len(doc.pictures)} found, {len(blocks)} captioned "
                f"({skipped_noise} noise, {skipped_unextractable} unextractable)")
    return blocks


def reingest_diagrams(file_path: str, figures_dir, embed_model, client, collection_name: str) -> list[dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    filename = os.path.basename(file_path)
    figures_dir = figures_dir or config.FIGURES_DIR

    logger.info(f"--- Re-ingesting diagrams only: {filename} ---")
    converter = DocumentConverter()
    doc = converter.convert(file_path).document

    diagram_blocks = process_diagrams(doc, file_path, filename, figures_dir)

    collection = client.get_or_create_collection(collection_name)
    collection.delete(where={"$and": [{"source_file": filename}, {"type": "diagram"}]})
    logger.info(f"  Deleted old diagram chunks for {filename}")

    if diagram_blocks:
        embed_and_store(diagram_blocks, embed_model, client, collection_name=collection_name)
    logger.info(f"--- Done: {len(diagram_blocks)} diagram chunk(s) re-ingested for {filename} ---")
    return diagram_blocks


def parse_heading(text: str, current_depth: int = 0) -> tuple[int, str, str]:
    text = text.strip()
    m = HEADING_NUM_RE.match(text)
    if m:
        numbering = m.group(1)
        title = m.group(2).strip()
        depth = numbering.count(".") + 1
        return depth, numbering, title
    return current_depth + 1, "", text

def extract_sections(doc) -> list[dict]:
    sections = []
    stack: list[tuple[int, str]] = []
    current_texts: list[str] = []
    current_page = None

    def flush():
        if current_texts:
            sections.append({
                "heading_path": " > ".join(t for _, t in stack),
                "page": current_page,
                "texts": list(current_texts),
            })

    for item, _ in doc.iterate_items():
        label = str(getattr(item, "label", "")).upper()
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        page_no = get_page_no(item)

        if "SECTION_HEADER" in label or "TITLE" in label:
            flush()
            current_texts = []
            current_top_depth = stack[-1][0] if stack else 0
            depth, numbering, title = parse_heading(text, current_depth=current_top_depth)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, f"{numbering} {title}".strip()))
            current_page = page_no
        elif "TEXT" in label or "PARAGRAPH" in label or "LIST_ITEM" in label:
            current_texts.append(text)
            if current_page is None:
                current_page = page_no

    flush()
    return sections

def _token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))

def split_into_token_chunks(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    units = []
    for para in paragraphs:
        if _token_len(para) <= chunk_size:
            units.append(para)
        else:
            for sent in re.split(r'(?<=[.!?])\s+', para):
                if sent.strip():
                    units.append(sent.strip())

    chunks, current, current_tokens = [], [], 0
    for unit in units:
        u_tokens = _token_len(unit)
        if current_tokens + u_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_units, overlap_tokens = [], 0
            for u in reversed(current):
                t = _token_len(u)
                if overlap_tokens + t > overlap:
                    break
                overlap_units.insert(0, u)
                overlap_tokens += t
            current, current_tokens = overlap_units, overlap_tokens
        current.append(unit)
        current_tokens += u_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks

def get_table_header(records: list[dict]) -> tuple[list[str], list[dict]]:
    if not records:
        return [], records
    keys = list(records[0].keys())
    if all(str(k).strip().isdigit() for k in keys):
        header_row = records[0]
        new_keys = [str(v) for v in header_row.values()]
        promoted = [dict(zip(new_keys, row.values())) for row in records[1:]]
        return new_keys, promoted
    return keys, records


def get_table_caption(doc, table, page_no: int) -> str:
    caption = table.caption_text(doc) if hasattr(table, "caption_text") else ""
    if caption:
        return caption
    for item, _ in doc.iterate_items():
        if "CAPTION" in str(getattr(item, "label", "")) and get_page_no(item) == page_no:
            return (getattr(item, "text", "") or "").strip()
    return ""

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

DOT_LEADER_RE = re.compile(r'\.\s*\.\s*\.\s*\.')

def is_navigational_table(headers: list[str], records: list[dict]) -> bool:
    combined = " ".join(headers)
    if DOT_LEADER_RE.search(combined):
        return True
    non_empty_headers = [h for h in headers if h.strip() and h.strip() != "-"]
    if non_empty_headers:
        duplicate_ratio = 1 - (len(set(non_empty_headers)) / len(non_empty_headers))
        if duplicate_ratio > 0.5:
            return True
    if records:
        sample = " ".join(str(v) for v in records[0].values())
        if DOT_LEADER_RE.search(sample):
            return True
    if len(records) <= 1:
        return True
    if headers and headers[0].strip() == "-":
        return True

    return False

def process_table(table, doc, filename: str) -> list[dict]:
    raw_records = table.export_to_dataframe(doc).to_dict(orient="records")
    headers, records = get_table_header(raw_records)
    if is_navigational_table(headers, records):
        logger.info(f"  Skipped navigational table (page {get_page_no(table)}) — looks like TOC/glossary, not data")
        return []
    page_no = get_page_no(table)
    caption = get_table_caption(doc, table, page_no)

    row_groups = [records[i:i + config.MAX_TABLE_ROWS_PER_CHUNK]
                  for i in range(0, len(records), config.MAX_TABLE_ROWS_PER_CHUNK)] or [[]]

    blocks = []
    for idx, group in enumerate(row_groups):
        sentence_version = " ".join(row_to_sentence(row) for row in group)
        caption_line = f"Table: {caption}. " if caption else ""
        header_line = f"Columns: {', '.join(headers)}. " if headers else ""
        part = f" (part {idx + 1}/{len(row_groups)})" if len(row_groups) > 1 else ""
        blocks.append({
            "content": f"{caption_line}{header_line}{sentence_version}",
            "type": "table",
            "page": page_no,
            "source_file": filename,
            "structured": json.dumps(group, ensure_ascii=False),
            "table_caption": caption,
        })
        _ = part
    return blocks

import fitz

def get_printed_page_number(pdf_path: str, physical_page_no: int) -> str | None:
    pdf = fitz.open(pdf_path)
    page = pdf[physical_page_no - 1]
    page_height = page.rect.height

    blocks = page.get_text("blocks")
    candidates = []
    for b in blocks:
        text = b[4].strip()
        y0 = b[1]
        if not text:
            continue
        if re.fullmatch(r"\d{1,4}", text) or re.fullmatch(r"[ivxlcdm]{1,6}", text.lower()):
            if y0 > page_height * 0.85:  # only the bottom ~15% of the page -- the footer zone
                candidates.append((y0, text))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)  # bottom-most wins
    return candidates[0][1]


def build_page_number_map(pdf_path: str) -> dict[int, str]:
    pdf = fitz.open(pdf_path)
    page_map = {}
    for physical in range(1, pdf.page_count + 1):
        label = get_printed_page_number(pdf_path, physical)
        if label:
            page_map[physical] = label
    return page_map


def build_printed_to_physical_map(page_map: dict[int, str]) -> dict[str, int]:
    reverse = {}
    for physical, label in page_map.items():
        if label.isdigit():
            reverse[label] = physical
    return reverse

def process_document(file_path: str, figures_dir: str = None) -> list[dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    figures_dir = figures_dir or config.FIGURES_DIR
    filename = os.path.basename(file_path)
    logger.info(f"--- Processing {filename} ---")
    start_time = time.time()
 
    converter = DocumentConverter()
    doc = converter.convert(file_path).document
    blocks = []
 
    #token-based text chunking
    sections = extract_sections(doc)
    for section in sections:
        for chunk_text in split_into_token_chunks(section["texts"], config.CHUNK_TOKEN_SIZE, config.CHUNK_TOKEN_OVERLAP):
            prefix = f"[{section['heading_path']}]\n" if section["heading_path"] else ""
            blocks.append({
                "content": prefix + chunk_text,
                "type": "text",
                "page": section["page"],
                "source_file": filename,
                "section": section["heading_path"],
            })
    logger.info(f"  {len(sections)} sections → {sum(1 for b in blocks if b['type'] == 'text')} text chunks")
 
    #Table-aware chunking
    for table in doc.tables:
        blocks.extend(process_table(table, doc, filename))
    logger.info(f"  Tables: {len(doc.tables)} → {sum(1 for b in blocks if b['type'] == 'table')} table chunks")
 
    #Diagram
    blocks.extend(process_diagrams(doc, file_path, filename, figures_dir))
 
    logger.info(f"--- Finished {filename} in {time.time() - start_time:.1f}s, {len(blocks)} blocks ---")
    return blocks
def embed_and_store(blocks: list[dict], embed_model, client, collection_name: str = None) -> None:
    # Embed every block's content and store it in ChromaDB with its metadata.
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
        if "section" in block:
            metadata["section"] = block["section"]
        if "table_caption" in block:
            metadata["table_caption"] = block["table_caption"]
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