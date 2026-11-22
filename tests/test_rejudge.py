import json

import evaluate
import evalkit
import rejudge
from evalkit import Question


def row(qid, qtype, passed, answer, refused, category="", trace=None):
    return {"id": qid, "type": qtype, "question": "?", "skipped": False, "passed": passed, "category": category,
            "reason": "", "refused": refused, "retrieval_ok": True, "seconds": 1.0, "answer": answer,
            "trace": trace or [{"source": "doc.docx", "decision": "kept"}], "warned": False}


def record():
    return {"time": "2026-11-10T10:00:00", "questions_file": "evaluation/q.csv", "model": "m", "settings": {},
            "summary": {"accuracy": 0.5, "passed": 1, "ran": 2},
            "results": [row("C023", "answerable", False, "I could not find that information in the documents provided.", True, "generation_refusal"),
                        row("C027", "answerable", False, "It was worth approximately US$63 million.", False, "generation_error"),
                        row("C001", "answerable", True, "7 November 2026", False)]}


QS = [Question("C023", "hedged", "Which provider?", [], ["did not identify"], [], ["doc.docx"]),
      Question("C027", "answerable", "How much?", [], ["10 billion yen", "63 million"], [], ["doc.docx"]),
      Question("C001", "answerable", "When?", ["7 November 2026"], [], [], ["doc.docx"])]


def test_rejudge_applies_the_corrected_questions_to_saved_answers():
    results, changes = rejudge.rejudge(record(), QS)
    assert [(c[0], c[1], c[2]) for c in changes] == [("C023", False, True), ("C027", False, True)]
    s = evalkit.summarize(results)
    assert s["passed"] == 3 and s["ran"] == 3
    assert results[0]["type"] == "hedged"


def test_rejudge_leaves_skipped_and_unknown_questions_alone():
    rec = record()
    rec["results"].append({"id": "ZZ", "type": "answerable", "skipped": True, "passed": None})
    rec["results"].append(row("GONE", "answerable", True, "x", False))   # question no longer in the file
    results, changes = rejudge.rejudge(rec, QS)
    assert results[-2]["skipped"] is True and results[-1]["passed"] is True
    assert len(changes) == 2


def test_find_run_picks_the_latest_run_of_the_same_question_file(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)
    (tmp_path / "run-20260101-000000-000001.json").write_text(json.dumps({"questions_file": "evaluation/q.csv"}))
    (tmp_path / "run-20260102-000000-000001.json").write_text(json.dumps({"questions_file": "evaluation/other.csv"}))
    (tmp_path / "run-20260103-000000-000001.json").write_text(json.dumps({"questions_file": "C:/x/evaluation/q.csv"}))
    assert rejudge.find_run("evaluation/q.csv").name == "run-20260103-000000-000001.json"
    assert rejudge.find_run("evaluation/missing.csv") is None
