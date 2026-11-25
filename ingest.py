"""ingest.py - read documents, chunk them, embed locally, store in ChromaDB."""
import re
from pathlib import Path

import chromadb
try:
    import pymupdf as fitz  # PyMuPDF (new name)
except ImportError:  # older installs
    import fitz
import ollama
from chromadb.config import Settings
from docx import Document

from settings import cfg

BASE = Path(__file__).parent
DOCS_DIR = BASE / "data" / "docs"
CHROMA_DIR = BASE / "data" / "chroma"

# Values come from settings.ini
EMBED_MODEL = cfg.embed_model
CHUNK_SIZE = cfg.chunk_size
OVERLAP = cfg.chunk_overlap
SECTION_CHARS = cfg.section_chars  # DOCX/TXT have no real pages, so we make "sections"

INDEX_VERSION = 0  # bumped whenever the index changes (lets rag.py rebuild its word index)

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


def _table_lines(table):
    """One line per table row. When the first row is a header, each row becomes
    'Term: Botnet; Meaning: A collection of ...' so the words keep their labels."""
    rows = []
    for row in table.rows:
        cells, previous = [], None
        for cell in row.cells:
            if cell._tc is previous:  # a horizontally merged cell is returned once per column it spans
                continue
            previous = cell._tc
            cells.append(" ".join(cell.text.split()))
        rows.append(cells)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if body and all(header) and len(set(header)) == len(header) and all(len(r) == len(header) for r in rows):
        return ["; ".join(f"{h}: {c}" for h, c in zip(header, r) if c) + "." for r in body]
    return [" | ".join(c for c in r if c) for r in rows]


def _docx_text(doc):
    """Paragraphs AND tables, in the order they appear in the document."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    lines = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            text = Paragraph(child, doc).text.strip()
            if text:
                lines.append(text)
        elif child.tag.endswith("}tbl"):
            lines.extend(l for l in _table_lines(Table(child, doc)) if l.strip())
    return "\n".join(lines)


_NUMBERED_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,2})+\s+[A-Z]")


def is_numbered_heading(text: str) -> bool:
    """'4.1 Phishing and social engineering' is a heading; '2.5 million users' and '1. Identify' are not."""
    t = text.strip()
    return len(t) <= 110 and bool(_NUMBERED_HEADING.match(t)) and not t.endswith((".", ":", ";", ","))


def _docx_units(doc):
    """Split a Word document into (heading, text) units at Heading 1, Heading 2 and numbered
    headings such as '7.5 Germany'. Many real documents type those numbers as plain text, so
    the style alone is not enough. Lower headings and tables stay inside their unit."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    units, state = [], {"heading": "", "lines": []}

    def flush():
        if state["lines"]:
            units.append((state["heading"], "\n".join(state["lines"])))
        state["lines"] = []

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            style = p.style.name if p.style is not None and p.style.name else ""
            if style.startswith("Heading 1") or style == "Title" or style.startswith("Heading 2") or is_numbered_heading(text):
                flush()
                state["heading"] = " ".join(text.split())[:120]
            else:
                state["lines"].append(text)
        elif child.tag.endswith("}tbl"):
            state["lines"].extend(l for l in _table_lines(Table(child, doc)) if l.strip())
    flush()
    return units


def extract_units(path):
    """Like extract_pages, but each unit also carries its section heading: (number, text, heading)."""
    path = Path(path)
    if path.suffix.lower() == ".docx" and cfg.heading_context:
        doc = Document(str(path))
        units = [(n, text, heading) for n, (heading, text) in enumerate(_docx_units(doc), 1) if text.strip()]
        if units:
            return units, "section"
    pages, unit = extract_pages(path)
    return [(n, text, "") for n, text in pages], unit


def extract_pages(path: Path):
    """Return (list of (number, text), unit_name). unit is 'page' or 'section'."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        with fitz.open(path) as pdf:
            pages = [(i + 1, page.get_text()) for i, page in enumerate(pdf)]
        return pages, "page"
    if ext == ".docx":
        doc = Document(str(path))
        return _split_sections(_docx_text(doc)), "section"
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
        next_start = max(end - overlap, start + 1)
        # start the next chunk at a word boundary, not in the middle of a word
        if text[next_start - 1] != " ":
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
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
    global INDEX_VERSION
    INDEX_VERSION += 1
    path = Path(path)
    collection = get_collection()
    collection.delete(where={"source": path.name})

    units, unit = extract_units(path)
    ids, docs, metas = [], [], []
    for number, text, heading in units:
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.name}:{number}:{j}")
            docs.append(f"[{heading}] {chunk}" if heading else chunk)  # the section title travels with every passage
            metas.append({"source": path.name, "page": number, "unit": unit, "heading": heading})

    if not docs:
        return 0  # e.g. a scanned PDF with no extractable text

    vectors = embed([f"{path.stem}: {d}" for d in docs], "search_document: ")
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
    global INDEX_VERSION
    INDEX_VERSION += 1
    get_collection().delete(where={"source": name})
    f = DOCS_DIR / name
    if f.exists():
        f.unlink()


def reindex_all(progress=None) -> int:
    """Rebuild the index from every file saved in data/docs."""
    files = [f for f in sorted(DOCS_DIR.glob("*")) if f.suffix.lower() in SUPPORTED]
    total = 0
    for i, f in enumerate(files, 1):
        total += index_file(f)
        if progress:
            progress(i, len(files), f.name)
    return total


if __name__ == "__main__":
    # Index everything already in data/docs:  python ingest.py
    for f in sorted(DOCS_DIR.glob("*")):
        if f.suffix.lower() in SUPPORTED:
            print(f"Indexing {f.name} ... {index_file(f)} chunks")
