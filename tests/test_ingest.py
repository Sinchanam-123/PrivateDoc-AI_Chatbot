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


# ---------- Word tables ----------
def _docx_with_table(tmp_path, header, rows, before="Intro paragraph.", after="Closing paragraph."):
    from docx import Document
    f = tmp_path / "t.docx"
    d = Document()
    d.add_paragraph(before)
    table = d.add_table(rows=1 + len(rows), cols=len(header))
    for i, h in enumerate(header):
        table.rows[0].cells[i].text = h
    for r, row in enumerate(rows, 1):
        for i, c in enumerate(row):
            table.rows[r].cells[i].text = c
    d.add_paragraph(after)
    d.save(f)
    return f


def test_docx_table_text_is_extracted_in_document_order(tmp_path):
    f = _docx_with_table(tmp_path, ["Term", "Meaning"],
                         [["Botnet", "A collection of compromised devices controlled by an operator"],
                          ["Firewall", "Filters network traffic according to defined rules"]])
    pages, unit = ingest.extract_pages(f)
    text = pages[0][1]
    assert "Term: Botnet; Meaning: A collection of compromised devices controlled by an operator." in text
    assert "Term: Firewall; Meaning: Filters network traffic" in text
    assert text.index("Intro paragraph.") < text.index("Botnet") < text.index("Closing paragraph.")


def test_docx_table_without_a_clean_header_falls_back_to_plain_rows(tmp_path):
    f = _docx_with_table(tmp_path, ["Same", "Same"], [["a", "b"]])
    text = ingest.extract_pages(f)[0][0][1]
    assert "Same | Same" in text and "a | b" in text


def test_docx_table_is_found_by_chunk_and_search_text(tmp_path):
    f = _docx_with_table(tmp_path, ["Term", "Meaning"], [["Attack surface", "The points through which a system may be exposed to attack"]])
    chunks = ingest.chunk_text(ingest.extract_pages(f)[0][0][1])
    assert any("exposed to attack" in c for c in chunks)


def test_docx_merged_cells_are_read_once_but_equal_neighbours_are_kept(tmp_path):
    from docx import Document
    f = tmp_path / "m.docx"
    d = Document()
    table = d.add_table(rows=2, cols=3)
    table.rows[0].cells[0].merge(table.rows[0].cells[1]).text = "Wide header"
    table.rows[0].cells[2].text = "Other"
    for i, v in enumerate(["Yes", "Yes", "No"]):
        table.rows[1].cells[i].text = v
    d.save(f)
    text = ingest.extract_pages(f)[0][0][1]
    assert text.count("Wide header") == 1
    assert "Yes | Yes | No" in text


# ---------- heading-aware Word documents ----------
def test_numbered_headings_are_recognised_but_sentences_are_not():
    assert ingest.is_numbered_heading("4.1 Phishing and social engineering")
    assert ingest.is_numbered_heading("7.5 Germany: Berlin administration ransomware incident")
    assert not ingest.is_numbered_heading("2.5 million users were affected")
    assert not ingest.is_numbered_heading("1. Identify the critical assets")
    assert not ingest.is_numbered_heading("4.2 This is a long sentence that ends like one.")
    assert not ingest.is_numbered_heading("Phishing")


def _docx_with_cases(tmp_path):
    from docx import Document
    f = tmp_path / "cases.docx"
    d = Document()
    d.add_heading("CHAPTER 7: CASES", level=1)
    d.add_paragraph("Introduction to the cases.")
    d.add_heading("7.1 Canvas learning platform", level=2)
    d.add_paragraph("Incident summary")
    d.add_paragraph("The attackers claimed a large data theft. No ransom amount was reported.")
    d.add_paragraph("7.2 Berlin administration")          # plain text heading, no Word style
    d.add_paragraph("Incident summary")
    d.add_paragraph("The attackers demanded 30 Bitcoin and exfiltrated 5.79 terabytes.")
    d.save(f)
    return f


def _headings_on(monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(ingest, "cfg", replace(ingest.cfg, heading_context=True))


def test_heading_context_is_off_by_default(tmp_path):
    units, _ = ingest.extract_units(_docx_with_cases(tmp_path))
    assert all(h == "" for _, _, h in units)


def test_docx_is_split_at_headings_including_plain_numbered_ones(tmp_path, monkeypatch):
    _headings_on(monkeypatch)
    units, unit = ingest.extract_units(_docx_with_cases(tmp_path))
    assert unit == "section"
    assert [h for _, _, h in units] == ["CHAPTER 7: CASES", "7.1 Canvas learning platform", "7.2 Berlin administration"]
    assert "30 Bitcoin" in units[2][1] and "30 Bitcoin" not in units[1][1]
    assert "Incident summary" in units[1][1]            # lower headings stay inline


def test_every_chunk_carries_its_section_title(library, monkeypatch):
    _headings_on(monkeypatch)
    f = library / "cases.docx"
    f.write_bytes(_docx_with_cases(library.parent).read_bytes())
    ingest.index_file(f)
    data = ingest.get_collection().get(include=["documents", "metadatas"])
    bitcoin = [(d, m) for d, m in zip(data["documents"], data["metadatas"]) if "30 Bitcoin" in d]
    assert len(bitcoin) == 1
    doc, meta = bitcoin[0]
    assert doc.startswith("[7.2 Berlin administration] ") and meta["heading"] == "7.2 Berlin administration"
    assert not any("Bitcoin" in d and "Canvas" in d.split("]")[0] for d in data["documents"])


def test_heading_context_can_be_switched_off(tmp_path, monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(ingest, "cfg", replace(ingest.cfg, heading_context=False))
    units, _ = ingest.extract_units(_docx_with_cases(tmp_path))
    assert all(h == "" for _, _, h in units)


def test_other_file_types_have_no_heading(library):
    f = library / "plain.txt"
    f.write_text("Just some plain text content here.", encoding="utf-8")
    ingest.index_file(f)
    meta = ingest.get_collection().get(include=["metadatas"])["metadatas"][0]
    assert meta["heading"] == ""
