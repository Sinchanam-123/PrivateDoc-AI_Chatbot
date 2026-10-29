"""qlog.py - local logs: retrieval scores per question, and thumbs up/down feedback.

Stays on this computer in logs/queries.jsonl. It never stores document text,
only file names, page numbers and scores. Turn it off in settings.ini.
"""
import json
from datetime import datetime
from pathlib import Path

from settings import cfg

LOG_FILE = Path(__file__).parent / "logs" / "queries.jsonl"
FEEDBACK_FILE = Path(__file__).parent / "logs" / "feedback.jsonl"


def log_query(question, trace, refused, seconds, path=LOG_FILE, s=cfg, warned=False):
    if not s.logging_enabled:
        return
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question if s.log_questions else None,
        "refused": bool(refused),
        "warned": bool(warned),
        "seconds": round(seconds, 2),
        "model": s.llm_model,
        "candidates": trace,
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the app


def log_feedback(verdict, question, answer, sources, refused, path=FEEDBACK_FILE, s=cfg):
    """Save a thumbs up ('up') or thumbs down ('down') on an answer, locally."""
    if not s.logging_enabled or verdict not in ("up", "down"):
        return
    keep = s.log_questions  # when text logging is off, no question or answer text is saved
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "question": question if keep else None,
        "answer": answer if keep else None,
        "refused": bool(refused),
        "sources": sources,
        "model": s.llm_model,
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_log(path=LOG_FILE):
    return _read_jsonl(path)


def read_feedback(path=FEEDBACK_FILE):
    return _read_jsonl(path)
