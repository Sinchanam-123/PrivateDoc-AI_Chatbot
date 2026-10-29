import json
from dataclasses import replace

import qlog
from settings import Settings

S = Settings()
TRACE = [{"source": "a.txt", "page": 1, "distance": 0.31, "overlap": 0.5, "decision": "kept"}]


def test_log_line_is_written(tmp_path):
    f = tmp_path / "logs" / "q.jsonl"
    qlog.log_query("what is x?", TRACE, refused=False, seconds=1.234, path=f, s=S)
    rec = json.loads(f.read_text().splitlines()[0])
    assert rec["question"] == "what is x?" and rec["refused"] is False
    assert rec["seconds"] == 1.23 and rec["candidates"] == TRACE
    assert "text" not in rec["candidates"][0]  # never stores document text


def test_logging_can_be_disabled(tmp_path):
    f = tmp_path / "q.jsonl"
    qlog.log_query("q", TRACE, True, 1, path=f, s=replace(S, logging_enabled=False))
    assert not f.exists()


def test_question_text_can_be_hidden(tmp_path):
    f = tmp_path / "q.jsonl"
    qlog.log_query("secret question", TRACE, True, 1, path=f, s=replace(S, log_questions=False))
    assert json.loads(f.read_text())["question"] is None
    assert "secret" not in f.read_text()


def test_read_log_skips_damaged_lines(tmp_path):
    f = tmp_path / "q.jsonl"
    qlog.log_query("one", TRACE, False, 1, path=f, s=S)
    with open(f, "a") as fh:
        fh.write("not json\n")
    qlog.log_query("two", TRACE, True, 1, path=f, s=S)
    assert [e["question"] for e in qlog.read_log(f)] == ["one", "two"]


def test_logging_failure_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    qlog.log_query("q", TRACE, False, 1, path=blocker / "sub" / "q.jsonl", s=S)  # parent is a file


# ---------- feedback ----------
def test_feedback_is_saved(tmp_path):
    f = tmp_path / "fb.jsonl"
    qlog.log_feedback("up", "what is x?", "It is y.", [{"source": "a.txt", "page": 1}], False, path=f, s=S)
    qlog.log_feedback("down", "q2", "a2", [], True, path=f, s=S)
    rows = qlog.read_feedback(f)
    assert [r["verdict"] for r in rows] == ["up", "down"]
    assert rows[0]["question"] == "what is x?" and rows[0]["sources"][0]["source"] == "a.txt"
    assert rows[1]["refused"] is True


def test_feedback_ignores_bad_verdicts_and_respects_privacy(tmp_path):
    f = tmp_path / "fb.jsonl"
    qlog.log_feedback("sideways", "q", "a", [], False, path=f, s=S)
    assert not f.exists()
    qlog.log_feedback("up", "secret q", "secret a", [], False, path=f, s=replace(S, log_questions=False))
    text = f.read_text()
    assert "secret" not in text
    qlog.log_feedback("up", "q", "a", [], False, path=tmp_path / "off.jsonl", s=replace(S, logging_enabled=False))
    assert not (tmp_path / "off.jsonl").exists()


def test_warning_is_recorded(tmp_path):
    f = tmp_path / "q.jsonl"
    qlog.log_query("q", TRACE, False, 1, path=f, s=S, warned=True)
    assert json.loads(f.read_text())["warned"] is True
