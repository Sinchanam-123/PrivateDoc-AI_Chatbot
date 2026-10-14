"""ingest.py - read documents, chunk them, embed locally, store in ChromaDB."""
import re
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import ollama
from chromadb.config import Settings
from docx import Document

BASE = Path(__file__).parent
DOCS_DIR = BASE / "data" / "docs"
CHROMA_DIR = BASE / "data" / "chroma"

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800      # characters per chunk (small, to fit a 2048-token context)
OVERLAP = 100         # characters shared between neighbouring chunks
SECTION_CHARS = 3000  # DOCX/TXT have no real pages, so we make "sections"

SUPPORTED = {".pdf", ".docx", ".txt"}


_collection = None


def get_collection():
    """Open ChromaDB once and reuse it (opening it every call is slow)."""
    global _collection
    if _collection is None:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            "documents", metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _split_sections(text: str):
    """Group plain text into pseudo-pages of roughly SECTION_CHARS characters."""
    text = text.strip()
    return [
        (i + 1, text[start:start + SECTION_CHARS])
        for i, start in enumerate(range(0, len(text), SECTION_CHARS))
    ]


def extract_pages(path: Path):
    """Return (list of (number, text), unit_name). unit is 'page' or 'section'."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        with fitz.open(path) as pdf:
            pages = [(i + 1, page.get_text()) for i, page in enumerate(pdf)]
        return pages, "page"
    if ext == ".docx":
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return _split_sections(text), "section"
    if ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return _split_sections(text), "section"
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to break on a space, not mid-word
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed(texts, prefix: str):
    """Embed texts with the local model. nomic-embed-text expects a task prefix."""
    vectors = []
    for i in range(0, len(texts), 16):
        batch = [prefix + t for t in texts[i:i + 16]]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"])
    return vectors


def index_file(path: Path) -> int:
    """Index one file (replacing any older version). Returns number of chunks."""
    path = Path(path)
    collection = get_collection()
    collection.delete(where={"source": path.name})

    pages, unit = extract_pages(path)
    ids, docs, metas = [], [], []
    for number, text in pages:
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.name}:{number}:{j}")
            docs.append(chunk)
            metas.append({"source": path.name, "page": number, "unit": unit})

    if not docs:
        return 0  # e.g. a scanned PDF with no extractable text

    vectors = embed(docs, "search_document: ")
    for i in range(0, len(docs), 64):
        collection.add(
            ids=ids[i:i + 64],
            documents=docs[i:i + 64],
            metadatas=metas[i:i + 64],
            embeddings=vectors[i:i + 64],
        )
    return len(docs)


def list_sources():
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def delete_source(name: str):
    get_collection().delete(where={"source": name})
    f = DOCS_DIR / name
    if f.exists():
        f.unlink()


if __name__ == "__main__":
    # Index everything already in data/docs:  python ingest.py
    for f in sorted(DOCS_DIR.glob("*")):
        if f.suffix.lower() in SUPPORTED:
            print(f"Indexing {f.name} ... {index_file(f)} chunks")