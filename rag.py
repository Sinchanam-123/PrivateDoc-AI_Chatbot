"""rag.py - retrieve chunks, check evidence, generate a grounded answer locally."""
import re
import threading

import ollama

from ingest import EMBED_MODEL, embed, get_collection

LLM_MODEL = "llama3.2:3b"   # switch to "llama3.2:1b" if the system feels sluggish
TOP_K = 4
MAX_DISTANCE = 0.60         # cosine distance; lower = stricter. Tune after testing.
NUM_CTX = 2048
KEEP_ALIVE = "15m"          # model unloads from memory after 15 idle minutes

REFUSAL = (
    "I could not find that information in the documents provided."
)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions from the user's documents. "
    "Combine the relevant information from ALL of the context passages into one "
    "clear, well-written answer of 2 to 5 sentences. Use only the context and "
    "never outside knowledge. Do not mention passages, numbers, or file names. "
    "If the context does not contain the answer, reply exactly: "
    "I could not find that information in the documents provided."
)


def retrieve(question: str):
    collection = get_collection()
    if collection.count() == 0:
        return []
    qvec = embed([question], "search_query: ")[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        if dist <= MAX_DISTANCE:  # evidence check
            hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def answer(question: str):
    """Return (answer_text, hits). hits is empty when we refuse."""
    hits = retrieve(question)
    if not hits:
        return REFUSAL, []

    context = "\n\n".join(h["text"] for h in hits)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
        options={"num_ctx": NUM_CTX, "temperature": 0.1, "num_predict": 300},
        keep_alive=KEEP_ALIVE,
    )
    text = response["message"]["content"].strip()

    if not re.search(r"[A-Za-z]{3,}", text):  # e.g. just "[2]"
        return REFUSAL, hits
    return text, hits


def unload_models():
    """Free RAM/VRAM immediately."""
    for model in (LLM_MODEL, "nomic-embed-text"):
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