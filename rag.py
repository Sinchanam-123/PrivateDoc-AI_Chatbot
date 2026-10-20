"""rag.py - retrieve, re-rank, check evidence, generate a grounded answer locally."""
import re
import threading

import ollama

from ingest import EMBED_MODEL, embed, get_collection
from settings import cfg

LLM_MODEL = cfg.llm_model
KEEP_ALIVE = cfg.keep_alive

REFUSAL = "I could not find that information in the documents provided."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided. Rules:\n"
    "1. First decide whether the context directly answers the question that was "
    "asked. If it does not (for example the question is general knowledge, or "
    "about something the context never discusses), reply with exactly this "
    "sentence and nothing else: " + REFUSAL + "\n"
    "2. Never answer a different question than the one asked, and never add "
    "loosely related facts.\n"
    "3. Match the length to the question. A simple fact (a colour, a name, a "
    "folder) gets 1 or 2 sentences. A broad question such as 'what is X', 'how "
    "does it work' or 'summarise' must get 3 to 5 sentences: say what it is, "
    "what it is for, and how it works, using every relevant passage and merging "
    "them into one smooth explanation.\n"
    "4. Copy names, numbers, dates and technical terms exactly as written.\n"
    "5. Never add facts that are not in the context, and never guess.\n"
    "6. Wrap the 2 or 3 most important key terms or facts in **double asterisks**.\n"
    "7. If the context covers only part of the question, answer that part and add "
    "one sentence saying what is not covered. Do not use the exact sentence from "
    "rule 1 in that case.\n"
    "8. Do not mention passages, numbers in brackets, or file names."
)

_STOP = {
    "the", "and", "for", "are", "was", "were", "what", "whats", "which", "who",
    "whom", "how", "why", "when", "where", "does", "did", "can", "could",
    "should", "would", "this", "that", "these", "those", "with", "from",
    "about", "into", "your", "you", "its", "has", "have", "had", "not", "any",
    "all", "tell", "give", "explain", "describe", "is", "of", "to", "in", "on",
    "my", "me", "it", "an", "as", "at", "be", "by", "do", "if", "or", "so",
    "we", "us", "up", "no", "am", "he",
}


def _keywords(text: str):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2 and w not in _STOP}


def select_hits(question, docs, metas, dists, s=cfg):
    """Apply the evidence checks and re-ranking to raw search results.

    Returns (hits, trace). `trace` has one entry per candidate saying what
    happened to it, which is written to the log to help with tuning.
    """
    q_words = _keywords(question)
    trace, pool = [], []
    for doc, meta, dist in zip(docs, metas, dists):
        overlap = len(q_words & _keywords(doc)) / len(q_words) if q_words else 0.0
        entry = {
            "source": meta.get("source"), "page": meta.get("page"),
            "distance": round(dist, 4), "overlap": round(overlap, 3), "decision": "kept",
        }
        trace.append(entry)
        if dist > s.max_distance:                       # check 1: close in meaning
            entry["decision"] = "too_far"
            continue
        if q_words and dist > s.strong_distance and overlap == 0:  # check 2: share a keyword
            entry["decision"] = "no_keyword"
            continue
        hit = {"text": doc, "meta": meta, "distance": dist,
               "score": (1 - dist) + s.keyword_weight * overlap}
        pool.append((entry, hit))
    if not pool:
        return [], trace

    best = min(h["distance"] for _, h in pool)
    kept = []
    for entry, hit in pool:                             # check 3: not much weaker than the best
        if hit["distance"] > best + s.distance_margin:
            entry["decision"] = "weak_vs_best"
        else:
            kept.append((entry, hit))
    kept.sort(key=lambda eh: eh[1]["score"], reverse=True)
    for entry, _ in kept[s.top_k:]:
        entry["decision"] = "below_top_k"
    return [h for _, h in kept[:s.top_k]], trace


def retrieve_with_trace(question: str):
    collection = get_collection()
    n = collection.count()
    if n == 0:
        return [], []
    qvec = embed([question], "search_query: ")[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=min(cfg.candidates, n),
        include=["documents", "metadatas", "distances"],
    )
    return select_hits(question, res["documents"][0], res["metadatas"][0], res["distances"][0])


def retrieve(question: str):
    return retrieve_with_trace(question)[0]


def prepare(question: str):
    """Return (hits, messages, trace). messages is None when there is no evidence."""
    hits, trace = retrieve_with_trace(question)
    if not hits:
        return [], None, trace
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    return hits, messages, trace


def stream_answer(messages):
    """Yield the answer piece by piece."""
    for part in ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"num_ctx": cfg.context_tokens, "temperature": cfg.temperature,
                 "num_predict": cfg.max_answer_tokens},
        keep_alive=KEEP_ALIVE,
        stream=True,
    ):
        yield part["message"]["content"]


def finalize(text: str, hits):
    """Clean up the model output. Returns (text, hits_to_cite)."""
    text = text.strip()
    if (
        not re.search(r"[A-Za-z]{3,}", text)
        or REFUSAL.lower() in text.lower()
        or text.lower().startswith("i could not find")
    ):
        return REFUSAL, []
    return text, hits


def unload_models():
    """Free RAM/VRAM immediately."""
    for model in (LLM_MODEL, EMBED_MODEL):
        try:
            if model == LLM_MODEL:
                ollama.generate(model=model, prompt="", keep_alive=0)
            else:
                ollama.embed(model=model, input="x", keep_alive=0)
        except Exception:
            pass


def warm_up():
    """Load both models in the background so the first question is fast."""
    def _run():
        try:
            ollama.embed(model=EMBED_MODEL, input="warm up", keep_alive=KEEP_ALIVE)
            ollama.generate(model=LLM_MODEL, prompt="", keep_alive=KEEP_ALIVE)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
