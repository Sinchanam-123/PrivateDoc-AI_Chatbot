r"""rejudge.py - score a finished run again with a corrected question file. No model needed.

If a question turns out to be badly written (for example it accepts only one of two correct
figures), fix the question file and run this. It reuses the answers already saved, so it takes
a second instead of a full evaluation run, and the result is saved as a normal run so that the
next `python evaluate.py` compares against the corrected baseline.

Usage (inside the venv):
    python rejudge.py --file evaluation\questions_cyber_80.csv
    python rejudge.py --file evaluation\questions_cyber_80.csv --run evaluation\results\run-....json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import evaluate
from evalkit import judge, load_questions, retrieval_ok, summarize


def find_run(questions_file, explicit=None):
    if explicit:
        return Path(explicit)
    for f in sorted(evaluate.RESULTS_DIR.glob("run-*.json"), reverse=True):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if Path(record.get("questions_file", "")).name == Path(questions_file).name:
            return f
    return None


def rejudge(record, questions):
    """Re-score every saved answer. Returns (new_results, changes) where changes lists
    (id, was_passed, now_passed, reason)."""
    by_id = {q.id: q for q in questions}
    results, changes = [], []
    for row in record["results"]:
        row = dict(row)
        q = by_id.get(row["id"])
        if q is None or row.get("skipped"):
            results.append(row)
            continue
        retrieved = [t.get("source") for t in row.get("trace", []) if t.get("decision") == "kept"]
        verdict = judge(q, row.get("answer", ""), bool(row.get("refused")), retrieved)
        if verdict.passed != row["passed"]:
            changes.append((row["id"], row["passed"], verdict.passed, verdict.reason))
        row.update(type=q.type, question=q.question, passed=verdict.passed, category=verdict.category,
                   reason=verdict.reason, retrieval_ok=retrieval_ok(q, retrieved))
        results.append(row)
    return results, changes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="the (corrected) question file")
    ap.add_argument("--run", help="a specific results .json (default: the latest run of that question file)")
    args = ap.parse_args()
    try:
        questions = load_questions(args.file)
    except (OSError, ValueError) as e:
        print(f"Problem with the question file: {e}")
        sys.exit(1)
    run = find_run(args.file, args.run)
    if not run or not run.exists():
        print("No saved run of that question file was found in evaluation/results.")
        sys.exit(1)
    record = json.loads(run.read_text(encoding="utf-8"))
    results, changes = rejudge(record, questions)
    old, new = record["summary"], summarize(results)
    print(f"Re-scoring {run.name} ({record['time']}) with {Path(args.file).name}\n")
    print(f"Accuracy before: {evaluate.pct(old['accuracy'])} ({old['passed']}/{old['ran']})")
    print(f"Accuracy now:    {evaluate.pct(new['accuracy'])} ({new['passed']}/{new['ran']})\n")
    if changes:
        print("Questions whose result changed because the question file changed:")
        for qid, was, now, reason in changes:
            print(f"  {qid}: {'PASS' if was else 'FAIL'} -> {'PASS' if now else 'FAIL'}   ({reason})")
    else:
        print("No question changed its result.")
    evaluate.print_summary(new)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = dict(record, time=datetime.now().isoformat(timespec="seconds"), summary=new, results=results,
               rejudged_from=record["time"])
    path = evaluate.RESULTS_DIR / f"run-{stamp}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved as {path.name}. The next evaluation run will compare against it.")


if __name__ == "__main__":
    main()
