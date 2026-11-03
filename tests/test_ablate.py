import pytest

import ablate
import evalkit
import evaluate
import rag
from evalkit import Question
from settings import Settings, apply_overrides, parse_overrides, validate


# ---------- --set parsing ----------
def test_parse_overrides_converts_types():
    got = parse_overrides(["top_k=3", "hybrid_search=false", "max_distance=0.5", "llm=ignored"][:3])
    assert got == {"top_k": 3, "hybrid_search": False, "max_distance": 0.5}
    assert parse_overrides(["polarity_check=ON", "sanitize_passages=0"]) == {"polarity_check": True, "sanitize_passages": False}
    assert parse_overrides(["llm_model=llama3.2:1b"]) == {"llm_model": "llama3.2:1b"}
    assert parse_overrides([]) == {} and parse_overrides(None) == {}


def test_parse_overrides_gives_readable_errors():
    with pytest.raises(ValueError, match="unknown setting"):
        parse_overrides(["topk=3"])
    with pytest.raises(ValueError, match="not a valid value"):
        parse_overrides(["top_k=many"])
    with pytest.raises(ValueError, match="not a valid value"):
        parse_overrides(["hybrid_search=maybe"])
    with pytest.raises(ValueError, match="name=value"):
        parse_overrides(["top_k"])


def test_apply_overrides_reports_bad_combinations():
    new, problems = apply_overrides(Settings(), {"top_k": 3})
    assert new.top_k == 3 and problems == []
    _, problems = apply_overrides(Settings(), {"top_k": 20})   # more than the 8 candidates
    assert problems


def test_every_shipped_variant_is_valid():
    for name, (description, overrides) in ablate.VARIANTS.items():
        assert description
        parse_overrides([f"{k}={v}" for k, v in overrides.items()])  # known names, right types
        _, problems = apply_overrides(Settings(), overrides)
        assert problems == [], name
    assert ablate.VARIANTS["baseline"][1] == {}


def test_settings_diff():
    assert evaluate.settings_diff({"top_k": 5, "a": 1}, {"top_k": 3, "a": 1}) == ["top_k: 5 -> 3"]
    assert evaluate.settings_diff({"top_k": 5}, {"top_k": 5}) == []


# ---------- the report ----------
def entry(name, passed_ids, all_ids, secs=2.0):
    results = [{"id": i, "type": "answerable", "passed": i in passed_ids, "skipped": False, "refused": False,
                "retrieval_ok": True, "seconds": secs, "category": "" if i in passed_ids else "generation_error"}
               for i in all_ids]
    return {"name": name, "description": f"{name} description", "results": results, "summary": evalkit.summarize(results)}


def test_ablation_report_shows_numbers_and_changed_questions():
    ids = ["Q1", "Q2", "Q3", "Q4"]
    report = evalkit.ablation_report([entry("baseline", {"Q1", "Q2", "Q3"}, ids), entry("no_hybrid", {"Q1", "Q4"}, ids)])
    assert "| baseline |" in report and "75% (3/4)" in report and "50% (2/4)" in report
    assert "**no_hybrid** is worse on: Q2, Q3" in report
    assert "**no_hybrid** is better on: Q4" in report
    assert "noise" in report


# ---------- the runner reaches the real settings and restores them ----------
QS = [Question("A1", "answerable", "q?", ["x"], [], [], ["doc.txt"]),
      Question("A2", "answerable", "q2?", ["x"], [], [], ["doc.txt"])]


def fake_ask_factory(seen):
    def fake_ask(question):
        seen.append((rag.cfg.hybrid_search, rag.cfg.top_k))
        # pretend the second question can only be answered when hybrid search is on
        ok = rag.cfg.hybrid_search or question == "q?"
        return {"answer": "x" if ok else "I could not find that information in the documents provided.",
                "raw": "", "refused": not ok, "warned": False, "retrieved": ["doc.txt"] if ok else [],
                "pages": [], "trace": [], "removed": 0, "seconds": 0.1}
    return fake_ask


def test_run_variants_applies_each_setting_and_restores(monkeypatch):
    seen = []
    monkeypatch.setattr(evaluate, "ask", fake_ask_factory(seen))
    before = rag.cfg
    entries = ablate.run_variants(["no_hybrid", "top_k_3"], QS, {"doc.txt"}, on_progress=lambda m: None)
    assert [e["name"] for e in entries] == ["baseline", "no_hybrid", "top_k_3"]
    by = {e["name"]: e["summary"] for e in entries}
    assert by["baseline"]["passed"] == 2 and by["top_k_3"]["passed"] == 2
    assert by["no_hybrid"]["passed"] == 1                    # the switch really reached the pipeline
    assert (False, 5) in seen and (True, 3) in seen
    assert rag.cfg is before                                  # settings restored


def test_run_variants_restores_settings_even_if_something_fails(monkeypatch):
    def boom(question):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(evaluate, "ask", boom)
    before = rag.cfg
    with pytest.raises(RuntimeError):
        ablate.run_variants(["no_hybrid"], QS, {"doc.txt"}, on_progress=lambda m: None)
    assert rag.cfg is before


def test_run_variants_skips_invalid_variants_and_saves_progress(monkeypatch):
    monkeypatch.setattr(evaluate, "ask", fake_ask_factory([]))
    monkeypatch.setitem(ablate.VARIANTS, "broken", ("impossible", {"top_k": 50}))
    messages, saved = [], []

    def progress(m):
        messages.append(m)
    progress.save = lambda entries: saved.append(len(entries))
    entries = ablate.run_variants(["broken", "no_hybrid"], QS, {"doc.txt"}, on_progress=progress)
    assert [e["name"] for e in entries] == ["baseline", "no_hybrid"]
    assert any("Skipping broken" in m for m in messages)
    assert saved == [1, 2]


def test_progress_callback_reports_every_question(monkeypatch):
    monkeypatch.setattr(evaluate, "ask", fake_ask_factory([]))
    seen = []
    ablate.run_variants(["no_hybrid"], QS, {"doc.txt"}, on_progress=lambda m: None,
                        on_question=lambda done, total, row: seen.append((done, total, row["id"])))
    assert seen == [(1, 2, "A1"), (2, 2, "A2"), (1, 2, "A1"), (2, 2, "A2")]   # baseline, then no_hybrid


def test_run_questions_progress_counts_skipped_questions_too():
    seen = []
    skipped = [Question("S1", "answerable", "q?", ["x"], [], [], ["missing.txt"])]
    evaluate.run_questions(skipped, set(), verbose=False, progress=lambda d, t, r: seen.append((d, t, r["skipped"])))
    assert seen == [(1, 1, True)]
