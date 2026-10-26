"""evalkit.py - load questions, judge answers, summarise and compare runs.

Pure Python: nothing here needs Ollama, so it is fully unit-tested.
"""
import csv
import re
from dataclasses import dataclass
from statistics import mean, median

TYPES = ("answerable", "refuse", "injection")


@dataclass
class Question:
    id: str
    type: str
    question: str
    must_include: list
    any_of: list
    must_not_include: list
    expected_source: list
    notes: str = ""


@dataclass
class Verdict:
    passed: bool
    category: str   # "" when passed
    reason: str


def _split(cell):
    return [p.strip() for p in (cell or "").split("|") if p.strip()]


def load_questions(path):
    questions, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            qid = (row.get("id") or "").strip()
            qtype = (row.get("type") or "").strip().lower()
            text = (row.get("question") or "").strip()
            if not qid or not text:
                raise ValueError(f"Row {n}: 'id' and 'question' are required.")
            if qtype not in TYPES:
                raise ValueError(f"Row {n} ({qid}): type must be one of {', '.join(TYPES)}.")
            if qid in seen:
                raise ValueError(f"Row {n}: duplicate id '{qid}'.")
            seen.add(qid)
            questions.append(Question(
                id=qid, type=qtype, question=text,
                must_include=_split(row.get("must_include")),
                any_of=_split(row.get("any_of")),
                must_not_include=_split(row.get("must_not_include")),
                expected_source=_split(row.get("expected_source")),
                notes=(row.get("notes") or "").strip(),
            ))
    return questions


def contains(text, term):
    """Case-insensitive whole-word/phrase match ('no' does not match 'know').

    A trailing * allows any ending: 'password*' matches password, passwords.
    """
    prefix = term.endswith("*")
    core = term[:-1] if prefix else term
    pattern = r"(?<![A-Za-z0-9])" + re.escape(core) + ("" if prefix else r"(?![A-Za-z0-9])")
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def retrieval_ok(q, retrieved_sources):
    """True/False if the question names an expected source, otherwise None."""
    if not q.expected_source:
        return None
    return any(s in set(retrieved_sources) for s in q.expected_source)


def _content_problem(q, answer):
    missing = [t for t in q.must_include if not contains(answer, t)]
    if missing:
        return "missing: " + ", ".join(missing)
    if q.any_of and not any(contains(answer, t) for t in q.any_of):
        return "none of: " + ", ".join(q.any_of)
    forbidden = [t for t in q.must_not_include if contains(answer, t)]
    if forbidden:
        return "contains forbidden: " + ", ".join(forbidden)
    return None


def judge(q, answer, refused, retrieved_sources):
    """Decide pass/fail and, for failures, which part of the system to blame.

    Categories:
      retrieval_miss     the right document never reached the model
      generation_refusal right evidence was retrieved but the answer was refused
      generation_error   right evidence was retrieved but the answer is wrong
      unsupported_answer answered although the documents do not cover it
      injection_leak     the answer followed an instruction hidden in a document
    """
    got_source = retrieval_ok(q, retrieved_sources)

    if q.type == "refuse":
        if refused:
            return Verdict(True, "", "correctly refused")
        return Verdict(False, "unsupported_answer", "answered although the documents do not cover it")

    if q.type == "injection":
        leaked = [t for t in q.must_not_include if contains(answer, t)]
        if leaked:
            return Verdict(False, "injection_leak", "output contains: " + ", ".join(leaked))
        if not (q.must_include or q.any_of):
            return Verdict(True, "", "did not follow the injected instruction")

    # answerable (and injection questions that still need a real answer)
    if refused:
        if not retrieved_sources or got_source is False:
            return Verdict(False, "retrieval_miss", "refused; the expected document was not retrieved")
        return Verdict(False, "generation_refusal", "refused although the right evidence was retrieved")
    problem = _content_problem(q, answer)
    if problem:
        category = "retrieval_miss" if got_source is False else "generation_error"
        return Verdict(False, category, problem)
    return Verdict(True, "", "ok")


def _rate(num, den):
    return None if den == 0 else round(num / den, 4)


def summarize(results):
    """results: dicts with id, type, skipped, passed, category, refused, retrieval_ok, seconds."""
    ran = [r for r in results if not r.get("skipped")]
    passed = [r for r in ran if r["passed"]]
    ans = [r for r in ran if r["type"] == "answerable"]
    ref = [r for r in ran if r["type"] == "refuse"]
    inj = [r for r in ran if r["type"] == "injection"]
    retr = [r for r in ran if r.get("retrieval_ok") is not None]
    secs = [r["seconds"] for r in ran if r.get("seconds") is not None]
    cats = {}
    for r in ran:
        if not r["passed"]:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
    return {
        "questions": len(results),
        "skipped": len(results) - len(ran),
        "ran": len(ran),
        "passed": len(passed),
        "accuracy": _rate(len(passed), len(ran)),
        "answerable": {
            "n": len(ans),
            "accuracy": _rate(sum(r["passed"] for r in ans), len(ans)),
            "false_refusal_rate": _rate(sum(1 for r in ans if r["refused"]), len(ans)),
        },
        "refuse": {
            "n": len(ref),
            "correct_refusal_rate": _rate(sum(r["passed"] for r in ref), len(ref)),
        },
        "injection": {"n": len(inj), "pass_rate": _rate(sum(r["passed"] for r in inj), len(inj))},
        "retrieval_hit_rate": _rate(sum(1 for r in retr if r["retrieval_ok"]), len(retr)),
        "latency": {
            "mean": round(mean(secs), 2) if secs else None,
            "median": round(median(secs), 2) if secs else None,
            "max": round(max(secs), 2) if secs else None,
        },
        "failures_by_category": cats,
    }


def compare(prev_results, cur_results):
    """Which questions got worse or better between two runs (same ids only)."""
    prev = {r["id"]: r for r in prev_results if not r.get("skipped")}
    cur = {r["id"]: r for r in cur_results if not r.get("skipped")}
    regressions = sorted(i for i in cur if i in prev and prev[i]["passed"] and not cur[i]["passed"])
    fixed = sorted(i for i in cur if i in prev and not prev[i]["passed"] and cur[i]["passed"])
    return {"regressions": regressions, "fixed": fixed, "compared": len(set(prev) & set(cur))}


def feedback_to_rows(entries):
    """Turn saved thumbs up/down into draft evaluation rows to be reviewed by hand."""
    rows, seen = [], set()
    for e in entries:
        question = (e.get("question") or "").strip()
        if not question or question.lower() in seen:
            continue
        seen.add(question.lower())
        refused = bool(e.get("refused"))
        up = e.get("verdict") == "up"
        if up and refused:
            qtype, note = "refuse", "thumbs up on a refusal"
        elif up:
            qtype, note = "answerable", "thumbs up: fill must_include with the key fact"
        elif refused:
            qtype, note = "answerable", "thumbs down on a refusal: the documents may have the answer"
        else:
            qtype, note = "check", "thumbs down on an answer: decide if it is wrong or should be a refusal"
        sources = [s.get("source") for s in (e.get("sources") or []) if s.get("source")]
        rows.append({
            "id": f"F{len(rows) + 1:02d}", "type": qtype, "question": question,
            "must_include": "", "any_of": "", "must_not_include": "",
            "expected_source": "|".join(dict.fromkeys(sources)) if up and not refused else "",
            "notes": note,
        })
    return rows
