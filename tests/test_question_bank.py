"""Keeps the question files honest: every expected keyword must really appear
in the documents the question points to."""
from pathlib import Path

import pytest

import evalkit
import ingest

EVAL = Path(__file__).parent.parent / "evaluation"
FILES = ["questions.csv", "questions_100.csv"]


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
