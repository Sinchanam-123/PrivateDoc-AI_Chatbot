"""feedback_to_eval.py - turn your thumbs up/down clicks into draft evaluation questions.

Usage (inside the venv):   python feedback_to_eval.py

Writes evaluation/candidates.csv. Open it in Excel, fill in must_include (the key
fact of a correct answer), fix the type if needed, then copy good rows into
evaluation/questions.csv. Rows with type 'check' need your decision.
"""
import csv
from pathlib import Path

from evalkit import feedback_to_rows
from qlog import read_feedback

OUT = Path(__file__).parent / "evaluation" / "candidates.csv"
COLUMNS = ["id", "type", "question", "must_include", "any_of", "must_not_include", "expected_source", "notes"]

entries = [e for e in read_feedback() if e.get("question")]
rows = feedback_to_rows(entries)
if not rows:
    print("No feedback with question text yet. Use the thumbs buttons under answers in the app first.")
else:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} draft questions to {OUT}")
