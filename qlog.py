"""qlog.py - local log of retrieval scores (one JSON line per question).

Stays on this computer in logs/queries.jsonl. It never stores document text,
only file names, page numbers and scores. Turn it off in settings.ini.
"""
import json
from datetime import datetime
from pathlib import Path

from settings import cfg

LOG_FILE = Path(__file__).parent / "logs" / "queries.jsonl"


def log_query(question, trace, refused, seconds, path=LOG_FILE, s=cfg):
    if not s.logging_enabled:
        return
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question if s.log_questions else None,
        "refused": bool(refused),
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


def read_log(path=LOG_FILE):
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
