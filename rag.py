"""rag.py - retrieve, re-rank, check evidence, generate a grounded answer locally."""
import re
import threading

import ollama

from ingest import EMBED_MODEL, embed, get_collection

LLM_MODEL = "llama3.2:3b"   # switch to "llama3.2:1b" if the system feels sluggish
CANDIDATES = 8              # passages fetched from the index
TOP_K = 5                   # passages given to the model after re-ranking
MAX_DISTANCE = 0.60         # cosine distance; lower = stricter
DISTANCE_MARGIN = 0.12      # drop passages much weaker than the best match
NUM_CTX = 3072             # a little more room so fuller answers fit
KEEP_ALIVE = "15m"

REFUSAL = "I could not find that information in the documents provided."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided. Rules:\n"
    "1. Match the length to the question. A simple fact gets 1 or 2 sentences. "
    "A broad question (what is X, how does it work, summarise) gets 3 to 6 "
    "sentences that cover every relevant detail from ALL the passages, merged "
    "into one smooth explanation rather than separate fragments.\n"
    "2. Copy names, numbers, dates and technical terms exactly as written.\n"
    "3. Never add facts that are not in the context, and never guess.\n"
    "4. Wrap the 2 or 3 most important key terms or facts in **double asterisks**.\n"
    "5. If the context covers only part of the question, answer that part and "
    "say which part is not covered.\n"
    "6. Do not mention passages, numbers in brackets, or file names.\n"
    "7. If the context does not contain the answer, reply exactly: "
    + REFUSAL
)

_STOP = {
    "the", "and", "for", "are", "was", "were", "what", "which", "who", "whom",
    "how", "why", "when", "where", "does", "did", "can", "could", "should",
    "would", "this", "that", "these", "those", "with", "from", "about", "into",
    "your", "you", "its", "has", "have", "had", "not", "any", "all", "tell",
    "give", "explain", "describe",
}


def _keywords(text: str):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def retrieve(question: str):
    collection = get_collection()
    n = collection.count()
    if n == 0:
        return []
    qvec = embed([question], "search_query: ")[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=min(CANDIDATES, n),
        include=["documents", "metadatas", "distances"],
    )
    q_words = _keywords(question)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        if dist > MAX_DISTANCE:  # evidence check
            continue
        overlap = len(q_words & _keywords(doc)) / len(q_words) if q_words else 0.0
        hits.append(
            {"text": doc, "meta": meta, "distance": dist, "score": (1 - dist) + 0.3 * overlap}
        )
    if not hits:
        return []
    best = min(h["distance"] for h in hits)
    hits = [h for h in hits if h["distance"] <= best + DISTANCE_MARGIN]
    hits.sort(key=lambda h: h["score"], reverse=True)  # vector + keyword re-rank
    return hits[:TOP_K]


def prepare(question: str):
    """Return (hits, messages). messages is None when there is no evidence."""
    hits = retrieve(question)
    if not hits:
        return [], None
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    return hits, messages


def stream_answer(messages):
    """Yield the answer piece by piece."""
    for part in ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 450},
        keep_alive=KEEP_ALIVE,
        stream=True,
    ):
        yield part["message"]["content"]


def finalize(text: str, hits):
    """Clean up the model output. Returns (text, hits_to_cite)."""
    text = text.strip()
    if not re.search(r"[A-Za-z]{3,}", text) or text.lower().startswith("i could not find"):
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