r"""ablate.py - measure what each feature contributes, by switching it off and re-running the questions.

Usage (inside the venv, with Ollama running and your documents indexed):
    python ablate.py --list                              show the variants
    python ablate.py --file evaluation\questions_100.csv run every variant on the 108 questions
    python ablate.py --names no_hybrid,top_k_3           run only some variants (baseline always runs)
    python ablate.py --type answerable                   only one question type (faster)

Each variant changes settings for that run only. Results are saved after every variant
(evaluation/results/ablation-*.json and .md), so stopping part-way loses nothing.
"""
import argparse
import json
import sys
import time
from datetime import datetime

import evaluate
import rag
from evalkit import ablation_report, summarize
from settings import apply_overrides

# name -> (what changes, settings overrides)
VARIANTS = {
    "baseline": ("Everything on (current settings)", {}),
    "no_hybrid": ("Meaning-based search only, no exact-word (BM25) search", {"hybrid_search": False}),
    "no_keyword_rerank": ("Rank by meaning only (keyword_weight = 0)", {"keyword_weight": 0.0}),
    "no_sentence_filter": ("Keep instruction-like sentences in passages", {"sanitize_passages": False}),
    "no_question_guard": ("Do not block 'ignore your rules' requests", {"block_override_requests": False}),
    "top_k_3": ("3 passages given to the model instead of 5", {"top_k": 3}),
    "top_k_8": ("8 passages given to the model instead of 5", {"top_k": 8}),
}


def run_variants(names, questions, indexed, on_progress=print, on_question=None):
    """Run the baseline plus the chosen variants. Returns a list of entries.
    The global settings are always restored afterwards, even if something fails."""
    original = rag.cfg
    order = ["baseline"] + [n for n in names if n != "baseline"]
    entries = []
    try:
        for name in order:
            description, overrides = VARIANTS[name]
            new_cfg, problems = apply_overrides(original, overrides)
            if problems:
                on_progress(f"Skipping {name}: {'; '.join(problems)}")
                continue
            rag.cfg = new_cfg
            on_progress(f"\n[{len(entries) + 1}/{len(order)}] {name}: {description}")
            started = time.time()
            results = evaluate.run_questions(questions, indexed, verbose=False, progress=on_question)
            summary = summarize(results)
            took = time.time() - started
            on_progress(f"    accuracy {summary['passed']}/{summary['ran']}  ({took / 60:.1f} min)")
            entries.append({"name": name, "description": description, "overrides": overrides,
                            "summary": summary, "results": results})
            saver = getattr(on_progress, "save", None)
            if saver:
                saver(entries)
    finally:
        rag.cfg = original
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(evaluate.DEFAULT_QUESTIONS))
    ap.add_argument("--names", help="comma-separated variant names (see --list)")
    ap.add_argument("--type", choices=["answerable", "refuse", "injection", "hedged"])
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, (desc, over) in VARIANTS.items():
            print(f"{name:<20} {desc}   {over or ''}")
        return
    names = [n.strip() for n in args.names.split(",")] if args.names else [n for n in VARIANTS if n != "baseline"]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        print(f"Unknown variant(s): {', '.join(unknown)}. Use --list to see the choices.")
        sys.exit(1)

    try:
        questions = evaluate.select_questions(args.file, args.type, args.only)
    except (OSError, ValueError) as e:
        print(f"Problem with the question file: {e}")
        sys.exit(1)
    if not questions:
        print("No questions to run.")
        sys.exit(1)
    indexed = evaluate.ready_or_exit(questions)

    n_variants = 1 + len([n for n in names if n != "baseline"])
    print(f"\nAblation: {n_variants} variants x {len(questions)} questions with {rag.cfg.llm_model}.")
    print("Each variant takes about as long as one normal evaluation run. You can stop at any time;")
    print("finished variants are already saved.\n")

    evaluate.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = evaluate.RESULTS_DIR / f"ablation-{stamp}.json"

    def progress(msg):
        print(msg, flush=True)

    def save(entries):
        json_path.write_text(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                         "questions_file": str(args.file), "entries": entries},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
    progress.save = save

    started_all = time.time()

    def counter(done, total, row):
        """Live line such as:  question 12/108   passed so far 11   elapsed 1.4 min"""
        passed = counter.passed = getattr(counter, "passed", 0) if done > 1 else 0
        if row.get("passed"):
            counter.passed += 1
        print(f"\r    question {done}/{total}   passed so far {counter.passed}   "
              f"total elapsed {(time.time() - started_all) / 60:.1f} min   ", end="", flush=True)
        if done == total:
            print()

    entries = run_variants(names, questions, indexed, progress, on_question=counter)
    if not entries:
        print("Nothing was run.")
        return
    report = ablation_report(entries)
    md_path = evaluate.RESULTS_DIR / f"ablation-{stamp}.md"
    md_path.write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved to {md_path.relative_to(evaluate.ROOT)} (paste the table into your report) and {json_path.name}")


if __name__ == "__main__":
    main()
