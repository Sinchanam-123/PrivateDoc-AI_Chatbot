"""app.py - Streamlit interface for Private Document AI (localhost only)."""
import html

import streamlit as st

from ingest import DOCS_DIR, SUPPORTED, delete_source, index_file, list_sources, reindex_all
from rag import REFUSAL, finalize, prepare, stream_answer, unload_models, warm_up

st.set_page_config(page_title="Private Document AI", page_icon="🔒", layout="centered")

# Fonts are system fonts on purpose: nothing is downloaded from the internet.
CSS = """
<style>
:root{--page:#ECEEF1;--sheet:#FFFFFF;--ink:#1C2733;--muted:#5B6773;--rule:#CDD3DA;--mark:#F7E27A;--blue:#24476B;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stBottom"] > div{background:var(--page);}
.block-container{max-width:760px;padding-top:2.2rem;padding-bottom:7rem;}
html, body, [data-testid="stAppViewContainer"]{font-family:"Segoe UI",system-ui,sans-serif;color:var(--ink);}
.app-title{font-family:Georgia,Cambria,"Times New Roman",serif;font-size:2rem;font-weight:600;letter-spacing:-0.01em;margin:0;line-height:1.2;}
.app-sub{color:var(--muted);font-size:.92rem;margin:.35rem 0 0;}
.q{font-weight:600;font-size:1.02rem;border-left:3px solid var(--blue);padding:.1rem 0 .1rem .8rem;margin:1.9rem 0 .7rem;}
.sheet{background:var(--sheet);border:1px solid var(--rule);border-radius:4px;padding:1.2rem 1.5rem;
  font-family:Georgia,Cambria,"Times New Roman",serif;font-size:1.06rem;line-height:1.7;color:var(--ink);}
.sheet.muted{color:var(--muted);font-style:italic;}
.sheet p{margin:0 0 .8rem;}
.sheet p:last-of-type{margin-bottom:0;}
.sources{margin-top:1.1rem;padding-top:.8rem;border-top:1px solid var(--rule);
  font-family:"Segoe UI",system-ui,sans-serif;font-size:.85rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:.35rem 1.2rem;align-items:baseline;}
.sources .lead{color:var(--ink);font-weight:600;}
.mark{background:linear-gradient(transparent 58%,var(--mark) 58%);color:var(--ink);padding:0 .12rem;}
.empty{color:var(--muted);margin-top:2.2rem;line-height:1.6;max-width:34rem;}
.stButton > button{border-radius:4px;border:1px solid var(--rule);background:#fff;font-weight:500;}
.stButton > button:hover{border-color:var(--blue);color:var(--blue);}
.stButton > button[kind="primary"]{background:var(--blue);border-color:var(--blue);color:#fff;}
[data-testid="stExpander"]{border:none;margin-top:.2rem;}
[data-testid="stExpander"] summary{font-size:.85rem;color:var(--muted);}
[data-testid="stChatInput"]{border-radius:4px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def _start():
    warm_up()  # background thread; does not block the page
    return True


_start()


@st.cache_data(show_spinner=False)
def cached_sources():
    return list_sources()


# ---------- Documents popup ----------
@st.dialog("Documents", width="large")
def documents_dialog():
    uploads = st.file_uploader(
        "Add PDF, Word or text files",
        type=[e.lstrip(".") for e in SUPPORTED],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded files", type="primary"):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            path = DOCS_DIR / up.name
            path.write_bytes(up.getbuffer())
            with st.spinner(f"Indexing {up.name}..."):
                n = index_file(path)
            if n:
                st.success(f"{up.name}: {n} passages indexed")
            else:
                st.warning(f"{up.name}: no text found (scanned PDF?)")
        cached_sources.clear()

    st.markdown("**Indexed documents**")
    sources = cached_sources()
    if not sources:
        st.info("Nothing indexed yet. Add a file above.")
    for name in sources:
        c1, c2 = st.columns([8, 1])
        c1.write(name)
        if c2.button("✕", key=f"del_{name}", help="Remove from index"):
            delete_source(name)
            cached_sources.clear()
            st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("Rebuild index", help="Re-read every saved document. Use after an update."):
        bar = st.progress(0.0)
        reindex_all(lambda i, n, name: bar.progress(i / n, text=f"Indexing {name}"))
        cached_sources.clear()
        st.success("Index rebuilt.")
    if c2.button("Unload model", help="Free memory now. It reloads on the next question."):
        unload_models()
        st.success("Model unloaded.")


# ---------- HTML helpers ----------
def question_html(q):
    return f'<div class="q">{html.escape(q)}</div>'


def sources_html(hits):
    seen, items = set(), []
    for h in hits:
        m = h["meta"]
        key = (m["source"], m["unit"], m["page"])
        if key in seen:
            continue
        seen.add(key)
        items.append(
            f'<span><span class="mark">{html.escape(m["source"])}</span> {m["unit"]} {m["page"]}</span>'
        )
    return '<div class="sources"><span class="lead">Sources</span>' + "".join(items) + "</div>"


def sheet_html(text, hits=None, muted=False):
    paras = "".join(
        f'<p>{html.escape(p.strip()).replace(chr(10), "<br>")}</p>'
        for p in text.split("\n\n")
        if p.strip()
    )
    cls = "sheet muted" if muted else "sheet"
    src = sources_html(hits) if hits else ""
    return f'<div class="{cls}">{paras}{src}</div>'


def passages_expander(hits):
    with st.expander("Show the passages used"):
        for h in hits:
            m = h["meta"]
            st.markdown(f"**{m['source']}, {m['unit']} {m['page']}**")
            st.caption(h["text"])


# ---------- Header ----------
sources_now = cached_sources()
left, right = st.columns([4, 1], vertical_alignment="center")
with left:
    count = len(sources_now)
    sub = (
        f"{count} document{'s' if count != 1 else ''} indexed. Everything runs on this computer."
        if count
        else "Everything runs on this computer."
    )
    st.markdown(
        f'<h1 class="app-title">Private Document AI</h1><p class="app-sub">{sub}</p>',
        unsafe_allow_html=True,
    )
with right:
    if st.button("Documents", use_container_width=True):
        documents_dialog()

# ---------- Conversation ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    msg = (
        "Ask a question. The answer is written only from your documents, and the pages it came from are listed underneath."
        if sources_now
        else "No documents yet. Open Documents to add a PDF, Word or text file."
    )
    st.markdown(f'<p class="empty">{msg}</p>', unsafe_allow_html=True)

for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(question_html(m["content"]), unsafe_allow_html=True)
    else:
        st.markdown(sheet_html(m["content"], m.get("hits")), unsafe_allow_html=True)
        if m.get("hits"):
            passages_expander(m["hits"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(question_html(question), unsafe_allow_html=True)

    slot = st.empty()
    slot.markdown(sheet_html("Searching your documents...", muted=True), unsafe_allow_html=True)
    try:
        hits, messages = prepare(question)
        if messages is None:
            text, final_hits = REFUSAL, []
        else:
            acc = ""
            for piece in stream_answer(messages):
                acc += piece
                slot.markdown(sheet_html(acc + " ▍"), unsafe_allow_html=True)
            text, final_hits = finalize(acc, hits)
    except Exception:
        text, final_hits = "Could not reach Ollama. Check that it is running, then ask again.", []

    slot.markdown(sheet_html(text, final_hits), unsafe_allow_html=True)
    if final_hits:
        passages_expander(final_hits)
    st.session_state.messages.append({"role": "assistant", "content": text, "hits": final_hits})