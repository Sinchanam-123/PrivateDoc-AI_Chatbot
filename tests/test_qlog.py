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
