"""Keeps the question files honest: every expected keyword must really appear
in the documents the question points to."""
from pathlib import Path

import pytest

import evalkit
import ingest

EVAL = Path(__file__).parent.parent / "evaluation"
FILES = ["questions.csv", "questions_100.csv", "questions_cyber_80.csv"]


def doc_text(name):
    for folder in (EVAL / "sample_docs", EVAL):
        path = folder / name
        if path.exists():
            pages, _ = ingest.extract_pages(path)
            return " ".join(text for _, text in pages)
    raise FileNotFoundError(name)


@pytest.mark.parametrize("filename", FILES)
def test_expected_keywords_exist_in_the_source_documents(filename):
    for q in evalkit.load_questions(EVAL / filename):
        if q.type == "refuse":
            continue
        if not q.expected_source:
            assert not (q.must_include or q.any_of), f"{q.id} needs an expected_source"
            continue
        text = " ".join(doc_text(s) for s in q.expected_source)
        for term in q.must_include:
            assert evalkit.contains(text, term), f"{q.id}: '{term}' is not in {q.expected_source}"
        if q.any_of:
            assert any(evalkit.contains(text, t) for t in q.any_of), f"{q.id}: none of {q.any_of} in sources"


def test_hundred_question_bank_shape():
    qs = evalkit.load_questions(EVAL / "questions_100.csv")
    assert 100 <= len(qs) <= 120
    assert len({q.id for q in qs}) == len(qs)
    texts = [q.question.lower() for q in qs]
    assert len(set(texts)) == len(texts), "duplicate questions inside the bank"
    kinds = {q.type for q in qs}
    assert kinds == {"answerable", "refuse", "injection"}


def test_bank_does_not_repeat_the_tuned_set():
    tuned = {q.question.lower() for q in evalkit.load_questions(EVAL / "questions.csv")}
    new = {q.question.lower() for q in evalkit.load_questions(EVAL / "questions_100.csv")}
    assert not (tuned & new)


def test_every_referenced_document_exists():
    for filename in FILES:
        for q in evalkit.load_questions(EVAL / filename):
            for source in q.expected_source:
                doc_text(source)


def test_cyber_question_bank_shape():
    qs = evalkit.load_questions(EVAL / "questions_cyber_80.csv")
    assert [q.id for q in qs] == [f"C{i:03d}" for i in range(1, 81)]
    assert len({q.question.lower() for q in qs}) == 80
    kinds = {q.type for q in qs}
    assert kinds == {"answerable", "refuse", "hedged"}
    assert {q.id for q in qs if q.type == "hedged"} == {"C023"}


def test_table_only_questions_can_be_answered_from_the_extracted_text():
    """The questions that depend on tables (glossary, NIST table, threat-actor table) must be
    answerable from what the app extracts, which proves the table reader works on the real document."""
    qs = {q.id: q for q in evalkit.load_questions(EVAL / "questions_cyber_80.csv")}
    text = doc_text("Cyberattacks_Around_the_World_Readable_Expanded_Assignment.docx")
    for qid in ("C016", "C017", "C048", "C054", "C055", "C056", "C057"):
        assert evalkit.terms_in_text(qs[qid], text), qid
