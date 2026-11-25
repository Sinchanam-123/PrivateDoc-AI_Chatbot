"""evaluate.py - score the chatbot against a set of questions with known answers.

Usage (inside the venv, with Ollama running):
    python evaluate.py                     run every question in evaluation/questions.csv
    python evaluate.py --type refuse       only one type (answerable, refuse, injection)
    python evaluate.py --only A01,A08      only these ids
    python evaluate.py --file my_set.csv   use another question file
    python evaluate.py --no-compare        do not compare with the previous run
    python evaluate.py --set top_k=3 --set hybrid_search=false
                                           change settings for this run only (not saved)

Each run is saved in evaluation/results/ (JSON for compare, CSV for Excel).
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from checks import run_checks
from evalkit import compare, judge, load_questions, retrieval_ok, summarize
from ingest import list_sources
import rag
from rag import generate_answer, prepare
from settings import apply_overrides, parse_overrides

ROOT = Path(__file__).parent
DEFAULT_QUESTIONS = ROOT / "evaluation" / "questions.csv"
RESULTS_DIR = ROOT / "evaluation" / "results"


def ask(question):
    """Run one question through the real pipeline. Returns the raw outcome."""
    started = time.time()
    hits, messages, trace = prepare(question)
    if messages is None:
        raw, answer, cited, warned = "", "", [], False
    else:
        answer, cited, raw, warned = generate_answer(question, hits, messages)
    return {
        "answer": answer,
        "raw": raw,
        "refused": not cited,
        "warned": warned,
        "retrieved": [h["meta"]["source"] for h in hits],
        "pages": [f'{h["meta"]["source"]} ({h["meta"]["unit"]} {h["meta"]["page"]})' for h in cited],
        "trace": trace,
        "passages": [{"source": h["meta"]["source"], "text": h["text"]} for h in hits],  # what the model was given
        "removed": sum(h.get("removed", 0) for h in hits),
        "seconds": round(time.time() - started, 2),
    }


def print_trace(trace, n=10):
    """Show the closest candidates and what the evidence checks did with them."""
    if not trace:
        return
    print("  passages considered (closest first):")
    for c in sorted(trace, key=lambda c: c.get("distance", 0))[:n]:
        print(f"    dist {c.get('distance', 0):.3f}  keywords {c.get('overlap', 0):.2f}  "
              f"{c.get('decision', ''):<24} {c.get('via', 'vector'):<8} {c.get('source')} ({c.get('page')})")


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def print_summary(s):
    print("\n" + "=" * 62)
    print(f"Questions run: {s['ran']} of {s['questions']}   (skipped: {s['skipped']})")
    print(f"Overall accuracy:          {pct(s['accuracy'])}   ({s['passed']}/{s['ran']})")
    print(f"Answerable questions:      {pct(s['answerable']['accuracy'])} correct   (n={s['answerable']['n']})")
    print(f"  false refusals:          {pct(s['answerable']['false_refusal_rate'])}")
    print(f"Questions to refuse:       {pct(s['refuse']['correct_refusal_rate'])} correctly refused   (n={s['refuse']['n']})")
    print(f"Prompt-injection tests:    {pct(s['injection']['pass_rate'])} passed   (n={s['injection']['n']})")
    print(f"Right document retrieved:  {pct(s['retrieval_hit_rate'])}")
    lat = s["latency"]
    print(f"Time per question:         mean {lat['mean']}s, median {lat['median']}s, max {lat['max']}s")
    if s["failures_by_category"]:
        print("Failures by cause:")
        for cat, n in sorted(s["failures_by_category"].items(), key=lambda kv: -kv[1]):
            print(f"  {cat:<20} {n}")
    print("=" * 62)


def load_previous(questions_file):
    """Latest earlier run of the SAME question file (so the comparison is meaningful)."""
    for f in sorted(RESULTS_DIR.glob("run-*.json"), reverse=True):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if Path(record.get("questions_file", "")).name == Path(questions_file).name:
            return record
    return None


def settings_diff(old, new):
    """Readable list of settings that differ between two runs."""
    return [f"{k}: {old.get(k)} -> {new.get(k)}" for k in sorted(new) if k in old and old[k] != new[k]]


def select_questions(path, qtype=None, only=None):
    questions = load_questions(path)
    if qtype:
        questions = [q for q in questions if q.type == qtype]
    if only:
        wanted = {i.strip() for i in only.split(",")}
        questions = [q for q in questions if q.id in wanted]
    return questions


def run_questions(questions, indexed, verbose=True, progress=None):
    """Ask every question and judge the answer. Returns the list of result rows.

    progress(done, total, row) is called after each question (used for a live counter)."""
    results = []
    for n, q in enumerate(questions, 1):
        row = {"id": q.id, "type": q.type, "question": q.question, "skipped": False}
        if q.type != "refuse" and q.expected_source and not (set(q.expected_source) & indexed):
            row.update(skipped=True, passed=None, category="", reason="document not indexed",
                       refused=None, retrieval_ok=None, seconds=None, answer="", sources=[],
                       raw_answer="", trace=[], sanitized=0, warned=False, passages=[])
            if verbose:
                print(f"SKIP  {q.id:<5} {q.question[:55]}")
            results.append(row)
            if progress:
                progress(n, len(questions), row)
            continue
        out = ask(q.question)
        verdict = judge(q, out["answer"], out["refused"], out["retrieved"])
        row.update(
            passed=verdict.passed, category=verdict.category, reason=verdict.reason,
            refused=out["refused"], retrieval_ok=retrieval_ok(q, out["retrieved"]),
            seconds=out["seconds"], answer=out["answer"], sources=out["pages"],
            raw_answer=out["raw"], trace=out["trace"], sanitized=out["removed"], warned=out["warned"],
            passages=out.get("passages", []),
        )
        results.append(row)
        if verbose:
            flag = "PASS" if verdict.passed else "FAIL"
            extra = "" if verdict.passed else f"  <- {verdict.category}: {verdict.reason}"
            note = "  [warned]" if out["warned"] else ""
            print(f"{flag}  {q.id:<5} {out['seconds']:>5.1f}s  {q.question[:50]}{note}{extra}")
        if progress:
            progress(n, len(questions), row)
    return results


def ready_or_exit(questions):
    """Check Ollama and the index. Returns the set of indexed documents, or exits with a message."""
    problems = run_checks()
    if problems:
        print("Cannot run the evaluation yet:")
        for p in problems:
            print(f"  - {p['title']}\n    {p['fix']}")
        sys.exit(1)
    indexed = set(list_sources())
    if not indexed:
        print("No documents are indexed. Add them in the app first (Documents button).")
        sys.exit(1)
    needed = {s for q in questions if q.type != "refuse" for s in q.expected_source}
    missing = sorted(s for s in needed if s not in indexed)
    if missing:
        print("These documents are not indexed, so questions that need only them will be skipped:")
        for m in missing:
            print(f"  - {m}")
    return indexed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(DEFAULT_QUESTIONS))
    ap.add_argument("--type", choices=["answerable", "refuse", "injection"])
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="change a setting for this run only, for example --set top_k=3")
    args = ap.parse_args()

    base_cfg = rag.cfg
    try:
        overrides = parse_overrides(args.set)
        if overrides:
            new_cfg, problems = apply_overrides(base_cfg, overrides)
            if problems:
                print("Those settings do not work together:")
                for p in problems:
                    print(f"  - {p}")
                sys.exit(1)
            rag.cfg = new_cfg
            print("Settings changed for this run only: " + ", ".join(f"{k}={v}" for k, v in overrides.items()))
    except ValueError as e:
        print(f"Problem with --set: {e}")
        sys.exit(1)

    try:
        questions = select_questions(args.file, args.type, args.only)
    except (OSError, ValueError) as e:
        print(f"Problem with the question file: {e}")
        sys.exit(1)
    if not questions:
        print("No questions to run.")
        sys.exit(1)
    indexed = ready_or_exit(questions)

    previous = None if args.no_compare else load_previous(args.file)
    print(f"\nRunning {len(questions)} questions with {rag.cfg.llm_model} ...\n")
    results = run_questions(questions, indexed)

    summary = summarize(results)
    print_summary(summary)
    warned = [r["id"] for r in results if r.get("warned")]
    if warned:
        bad = [r["id"] for r in results if r.get("warned") and not r.get("skipped") and not r["passed"]]
        print(f"Answers shown with a 'check this answer' warning: {len(warned)} ({', '.join(warned)})")
        print(f"  of the failures, {summary['failures_warned']} carried the warning ({', '.join(bad) or 'none'})")

    failures = [r for r in results if not r.get("skipped") and not r["passed"]]
    if failures:
        print("\nFAILED QUESTIONS (what the chatbot said):")
        for r in failures:
            shown = (r["answer"] or "(no answer)").replace("\n", " ")
            print(f"\n[{r['id']}] {r['question']}\n  cause: {r['category']} - {r['reason']}\n  answer: {shown[:300]}")
            raw = (r.get("raw_answer") or "").replace("\n", " ")
            if raw and raw != shown:
                print(f"  model wrote: {raw[:300]}")
            if r["category"] in ("retrieval_miss", "generation_refusal", "unsupported_answer"):
                print_trace(r.get("trace"))

    settings_snapshot = {k: v for k, v in asdict(rag.cfg).items() if k != "warnings"}
    if previous:
        diff = compare(previous["results"], results)
        print(f"\nCompared with the previous run ({previous['time']}), {diff['compared']} questions in common:")
        changed = settings_diff(previous.get("settings", {}), settings_snapshot)
        if changed:
            print("  settings differ: " + "; ".join(changed))
        print(f"  newly failing: {', '.join(diff['regressions']) or 'none'}")
        print(f"  newly fixed:   {', '.join(diff['fixed']) or 'none'}")
        old = previous["summary"].get("accuracy")
        if old is not None and summary["accuracy"] is not None:
            delta = (summary["accuracy"] - old) * 100
            print(f"  accuracy: {pct(old)} -> {pct(summary['accuracy'])} ({delta:+.0f} points)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    record = {"time": datetime.now().isoformat(timespec="seconds"), "model": rag.cfg.llm_model,
              "settings": settings_snapshot, "questions_file": str(args.file),
              "summary": summary, "results": results}
    (RESULTS_DIR / f"run-{stamp}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(RESULTS_DIR / f"run-{stamp}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "result", "cause", "reason", "refused",
                    "right_document_retrieved", "seconds", "answer", "model_output", "sources", "sanitized_sentences", "warned"])
        for r in results:
            result = "SKIP" if r.get("skipped") else ("PASS" if r["passed"] else "FAIL")
            w.writerow([r["id"], r["type"], r["question"], result, r["category"], r["reason"], r["refused"],
                        r["retrieval_ok"], r["seconds"], r["answer"], r.get("raw_answer", ""),
                        "; ".join(r["sources"]), r.get("sanitized", 0), r.get("warned", False)])
    print(f"\nSaved to evaluation/results/run-{stamp}.json and .csv")


if __name__ == "__main__":
    main()
