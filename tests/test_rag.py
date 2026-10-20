from dataclasses import replace

import rag
from settings import Settings

S = Settings()  # max 0.55, strong 0.40, margin 0.12, top_k 5


def meta(name="doc.txt", page=1):
    return {"source": name, "page": page, "unit": "section"}


def run(question, rows, s=S):
    """rows = [(text, distance), ...]"""
    docs = [r[0] for r in rows]
    metas = [meta(f"d{i}.txt") for i in range(len(rows))]
    dists = [r[1] for r in rows]
    return rag.select_hits(question, docs, metas, dists, s)


def decisions(trace):
    return [t["decision"] for t in trace]


# ---------- keywords ----------
def test_keywords_ignore_filler_but_keep_short_terms():
    assert rag._keywords("what is ai??") == {"ai"}
    assert rag._keywords("whats my name") == {"name"}
    assert rag._keywords("What is the fictional accent colour?") == {"fictional", "accent", "colour"}


# ---------- evidence checks ----------
def test_far_chunks_are_rejected():
    hits, trace = run("qwompel folder", [("qwompel folder text", 0.9)])
    assert hits == [] and decisions(trace) == ["too_far"]


def test_weak_match_needs_a_shared_keyword():
    hits, trace = run("what is ai", [("unrelated text about chatbots", 0.50)])
    assert hits == [] and decisions(trace) == ["no_keyword"]  # the earlier "what is AI?" bug


def test_weak_match_with_keyword_is_kept():
    hits, _ = run("what is ai", [("this explains ai in detail", 0.50)])
    assert len(hits) == 1


def test_strong_match_is_kept_without_keyword():
    hits, trace = run("what is ai", [("something close in meaning", 0.30)])
    assert len(hits) == 1 and decisions(trace) == ["kept"]


def test_question_of_only_filler_words_relies_on_distance():
    hits, _ = run("what is", [("anything", 0.50)])
    assert len(hits) == 1


def test_chunks_much_weaker_than_best_are_dropped():
    hits, trace = run("colour accent", [("colour accent teal", 0.30), ("colour accent other", 0.45)])
    assert len(hits) == 1
    assert decisions(trace) == ["kept", "weak_vs_best"]


# ---------- ranking ----------
def test_keyword_overlap_can_reorder_results():
    hits, _ = run("teal accent colour", [("generic words", 0.30), ("the teal accent colour", 0.35)])
    assert hits[0]["text"] == "the teal accent colour"


def test_top_k_limit_and_trace_marks_the_rest():
    rows = [(f"colour text {i}", 0.30 + i * 0.005) for i in range(8)]
    hits, trace = run("colour", rows, replace(S, top_k=3))
    assert len(hits) == 3
    assert decisions(trace).count("below_top_k") == 5


def test_trace_has_one_entry_per_candidate():
    hits, trace = run("colour", [("colour a", 0.3), ("zzz", 0.9), ("colour b", 0.31)])
    assert len(trace) == 3
    assert {"source", "page", "distance", "overlap", "decision"} <= set(trace[0])


# ---------- finalize ----------
def test_finalize_keeps_a_normal_answer():
    text, hits = rag.finalize("  The accent colour is **teal**.  ", ["h"])
    assert text == "The accent colour is **teal**." and hits == ["h"]


def test_finalize_refusal_anywhere_discards_everything():
    mixed = "Some loosely related facts. " + rag.REFUSAL
    assert rag.finalize(mixed, ["h"]) == (rag.REFUSAL, [])


def test_finalize_rejects_junk_and_refusal_prefix():
    assert rag.finalize("[2]", ["h"]) == (rag.REFUSAL, [])
    assert rag.finalize("I could not find it, sorry.", ["h"]) == (rag.REFUSAL, [])


# ---------- prepare ----------
def test_prepare_without_evidence_makes_no_prompt(monkeypatch):
    monkeypatch.setattr(rag, "retrieve_with_trace", lambda q: ([], [{"decision": "too_far"}]))
    hits, messages, trace = rag.prepare("anything")
    assert hits == [] and messages is None and trace


def test_prepare_builds_prompt_from_hits(monkeypatch):
    fake = [{"text": "Teal is the accent.", "meta": meta(), "distance": 0.2, "score": 0.9}]
    monkeypatch.setattr(rag, "retrieve_with_trace", lambda q: (fake, []))
    hits, messages, _ = rag.prepare("What colour?")
    assert messages[0]["role"] == "system"
    assert "Teal is the accent." in messages[1]["content"]
    assert "What colour?" in messages[1]["content"]
