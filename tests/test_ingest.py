import re
import zlib

import pytest

import ingest


# ---------- chunking ----------
def test_chunk_text_empty_and_short():
    assert ingest.chunk_text("") == []
    assert ingest.chunk_text("   \n  ") == []
    assert ingest.chunk_text("A short sentence.") == ["A short sentence."]


def test_chunks_respect_size_and_do_not_cut_words():
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words)
    chunks = ingest.chunk_text(text, size=200, overlap=30)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c) <= 200
    seen = set()
    for c in chunks:
        for w in c.split():
            assert w in words  # no half words
            seen.add(w)
    assert seen == set(words)  # nothing lost


def test_neighbouring_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = ingest.chunk_text(text, size=150, overlap=40)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split())


def test_whitespace_is_normalised():
    assert ingest.chunk_text("a   b\n\n c") == ["a b c"]


# ---------- sections ----------
def test_split_sections_numbers_from_one():
    secs = ingest._split_sections("x" * (ingest.SECTION_CHARS * 2 + 10))
    assert [n for n, _ in secs] == [1, 2, 3]


# ---------- extraction ----------
def test_extract_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    pages, unit = ingest.extract_pages(f)
    assert unit == "section" and pages == [(1, "hello world")]


def test_extract_docx(tmp_path):
    from docx import Document
    f = tmp_path / "a.docx"
    d = Document()
    d.add_paragraph("First paragraph.")
    d.add_paragraph("Second paragraph.")
    d.save(f)
    pages, unit = ingest.extract_pages(f)
    assert unit == "section"
    assert "First paragraph." in pages[0][1] and "Second paragraph." in pages[0][1]


def test_extract_pdf_uses_real_page_numbers(tmp_path):
    f = tmp_path / "a.pdf"
    doc = ingest.fitz.open()
    for text in ("page one text", "page two text"):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(f)
    doc.close()
    pages, unit = ingest.extract_pages(f)
    assert unit == "page"
    assert [n for n, _ in pages] == [1, 2]
    assert "two" in pages[1][1]


def test_unsupported_type_raises(tmp_path):
    f = tmp_path / "a.xyz"
    f.write_text("x")
    with pytest.raises(ValueError):
        ingest.extract_pages(f)


# ---------- indexing end to end (fake embeddings, temporary database) ----------
def fake_embed(texts, prefix):
    out = []
    for t in texts:
        v = [0.0] * 16
        for w in re.findall(r"[a-z0-9]+", (prefix + t).lower()):
            v[zlib.crc32(w.encode()) % 16] += 1.0
        out.append(v)
    return out


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(ingest, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(ingest, "_collection", None)
    monkeypatch.setattr(ingest, "embed", fake_embed)
    (tmp_path / "docs").mkdir()
    return tmp_path / "docs"


def test_index_list_reindex_delete(library):
    f = library / "notes.txt"
    f.write_text("Alpha facts. " * 120, encoding="utf-8")

    n = ingest.index_file(f)
    assert n > 1
    assert ingest.list_sources() == ["notes.txt"]
    count = ingest.get_collection().count()

    ingest.index_file(f)  # re-indexing must not create duplicates
    assert ingest.get_collection().count() == count

    ingest.delete_source("notes.txt")
    assert ingest.list_sources() == []
    assert not f.exists()


def test_file_without_text_indexes_nothing(library):
    f = library / "empty.txt"
    f.write_text("   ", encoding="utf-8")
    assert ingest.index_file(f) == 0


def test_metadata_has_source_and_page(library):
    f = library / "meta.txt"
    f.write_text("Some searchable content here.", encoding="utf-8")
    ingest.index_file(f)
    data = ingest.get_collection().get(include=["metadatas"])
    m = data["metadatas"][0]
    assert m["source"] == "meta.txt" and m["page"] == 1 and m["unit"] == "section"


def test_reindex_all_rebuilds_from_saved_files(library):
    (library / "a.txt").write_text("Alpha content " * 20, encoding="utf-8")
    (library / "b.txt").write_text("Beta content " * 20, encoding="utf-8")
    seen = []
    total = ingest.reindex_all(lambda i, n, name: seen.append((i, n, name)))
    assert total > 0
    assert sorted(ingest.list_sources()) == ["a.txt", "b.txt"]
    assert [s[1] for s in seen] == [2, 2]


# ---------- tables in Word documents ----------
def _docx_with_tables(path):
    from docx import Document
    d = Document()
    d.add_paragraph("Before the tables.")
    t = d.add_table(rows=3, cols=2)                     # glossary style: two columns
    for r, (a, b) in enumerate([("Term", "Meaning"), ("Botnet", "Compromised devices run by an operator."),
                                ("Firewall", "Filters network traffic.")]):
        t.cell(r, 0).text, t.cell(r, 1).text = a, b
    d.add_paragraph("Between the tables.")
    t2 = d.add_table(rows=3, cols=3)                    # header row with three columns
    for r, row in enumerate([("Actor", "Goal", "Defence"), ("Hacktivists", "Political message", "Resilience"),
                             ("Insiders", "Misuse of access", "Least privilege")]):
        for c, text in enumerate(row):
            t2.cell(r, c).text = text
    t3 = d.add_table(rows=2, cols=3)                    # first row is not a header: a merged title row
    t3.cell(0, 0).merge(t3.cell(0, 2)).text = "A merged title"
    for c, text in enumerate(("alpha", "beta", "gamma")):
        t3.cell(1, c).text = text
    d.add_paragraph("After the tables.")
    d.save(path)


def test_docx_tables_are_extracted_in_document_order(tmp_path):
    f = tmp_path / "t.docx"
    _docx_with_tables(f)
    pages, unit = ingest.extract_pages(f)
    text = "\n".join(t for _, t in pages)
    assert unit == "section"
    lines = text.split("\n")
    assert lines[0] == "Before the tables."
    assert "Botnet: Compromised devices run by an operator." in lines
    assert "Firewall: Filters network traffic." in lines
    assert "Actor: Hacktivists; Goal: Political message; Defence: Resilience" in lines
    assert "Actor: Insiders; Goal: Misuse of access; Defence: Least privilege" in lines
    assert text.index("Before") < text.index("Botnet") < text.index("Between") < text.index("Hacktivists") < text.index("After")


def test_merged_cells_do_not_repeat_and_headerless_rows_fall_back_to_pipes(tmp_path):
    f = tmp_path / "t.docx"
    _docx_with_tables(f)
    text = "\n".join(t for _, t in ingest.extract_pages(f)[0])
    assert text.count("A merged title") == 1
    assert "alpha | beta | gamma" in text.split("\n")


def test_empty_tables_and_blank_rows_add_nothing(tmp_path):
    from docx import Document
    d = Document()
    d.add_paragraph("Only text.")
    d.add_table(rows=2, cols=2)
    f = tmp_path / "e.docx"
    d.save(f)
    assert ingest.extract_pages(f)[0] == [(1, "Only text.")]


def test_table_rows_are_searchable_after_indexing(library):
    from docx import Document
    d = Document()
    d.add_paragraph("A short document.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Term", "Meaning"
    t.cell(1, 0).text, t.cell(1, 1).text = "Zeppelin", "A rigid airship."
    f = library / "g.docx"
    d.save(f)
    ingest.index_file(f)
    data = ingest.get_collection().get(include=["documents"])
    assert any("Zeppelin: A rigid airship." in doc for doc in data["documents"])
