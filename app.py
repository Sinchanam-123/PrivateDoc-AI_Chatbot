"""app.py - Streamlit interface for Private Document AI (localhost only)."""
import html
import re
import time

import streamlit as st

from checks import run_checks
from ingest import DOCS_DIR, SUPPORTED, delete_source, index_file, list_sources, reindex_all
from qlog import log_feedback, log_query
from rag import REFUSAL, finalize, prepare, stream_answer, unload_models, warm_up

st.set_page_config(page_title="Private Document AI", page_icon="🔒", layout="centered")

# Fonts are system fonts on purpose: nothing is downloaded from the internet.
CSS = """
<style>
:root{--page:#2F45D3;--ink:#0F1633;--muted:#4A5170;--mark:#FFE45C;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none !important;}
.stApp, [data-testid="stAppViewContainer"]{background:var(--page);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stBottom"], [data-testid="stBottom"] > div{background:var(--page);}
.block-container{max-width:780px;padding-top:2.4rem;padding-bottom:7rem;}
html, body, [data-testid="stAppViewContainer"]{font-family:"Segoe UI",system-ui,sans-serif;color:var(--ink);}
::selection{background:var(--mark);color:var(--ink);}

.block-container h1.app-title{font-family:Georgia,Cambria,"Times New Roman",serif !important;font-size:2.6rem !important;
  font-weight:700 !important;letter-spacing:-0.015em;line-height:1.12 !important;margin:0 !important;padding:0 !important;color:#fff !important;}
.app-sub{color:rgba(255,255,255,.84);font-size:.93rem;margin:.5rem 0 0;}
.rule{border-bottom:3px double rgba(255,255,255,.55);margin:1rem 0 0;}

.q{display:block;width:fit-content;max-width:85%;margin:2.1rem 0 1.1rem auto;background:var(--mark);color:var(--ink);
  font-weight:600;font-size:1.02rem;padding:.55rem .95rem;border:2px solid var(--ink);border-radius:4px;box-shadow:3px 3px 0 var(--ink);}
.sheet{background:#fff;border:2px solid var(--ink);border-radius:4px;padding:1.3rem 1.6rem;margin-bottom:1.2rem;
  box-shadow:5px 5px 0 var(--ink);font-family:Georgia,Cambria,"Times New Roman",serif;font-size:1.08rem;line-height:1.72;color:var(--ink);}
.sheet.muted{color:var(--muted);font-style:italic;}
.sheet p{margin:0 0 .85rem;}
.sheet p:last-child{margin-bottom:0;}
.sheet ul{margin:0 0 .85rem 1.1rem;padding:0;}
.sheet li{margin-bottom:.3rem;}
.sheet strong{font-weight:600;background:linear-gradient(transparent 55%,var(--mark) 55%);}
.mark{background:linear-gradient(transparent 55%,var(--mark) 55%);color:var(--ink);padding:0 .12rem;font-weight:600;}
.empty{color:rgba(255,255,255,.88);margin-top:2.4rem;line-height:1.65;max-width:34rem;font-size:1.02rem;}

.stButton > button, [data-testid="stPopoverButton"]{background:#fff;color:var(--ink);font-weight:600;border:2px solid var(--ink);
  border-radius:4px;box-shadow:3px 3px 0 var(--ink);transition:transform .08s, box-shadow .08s;}
.stButton > button:hover, [data-testid="stPopoverButton"]:hover{background:#fff;color:var(--ink);border-color:var(--ink);
  transform:translate(-1px,-1px);box-shadow:4px 4px 0 var(--ink);}
.stButton > button:active, [data-testid="stPopoverButton"]:active{transform:translate(2px,2px);box-shadow:1px 1px 0 var(--ink);}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{background:var(--mark);color:var(--ink);border:2px solid var(--ink);}
[data-testid="stPopoverButton"]{min-height:2.1rem;padding:.1rem .9rem;font-size:.88rem;}
[data-testid="stPopoverBody"]{border:2px solid var(--ink);border-radius:4px;}
div[role="dialog"]{border:2px solid var(--ink);border-radius:6px;}

[data-testid="stChatInput"]{background:#fff;border:2px solid var(--ink);border-radius:4px;box-shadow:4px 4px 0 var(--ink);}
[data-testid="stChatInput"] > div{border:none !important;box-shadow:none !important;}
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
_BULLET = re.compile(r"^\s*[-*•]\s+")


def _inline(s):
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(s))


def _format_block(block):
    out, bullets, text = [], [], []

    def flush_text():
        if text:
            out.append("<p>" + "<br>".join(text) + "</p>")
            text.clear()

    def flush_bullets():
        if bullets:
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in block.split("\n"):
        if not line.strip():
            continue
        if _BULLET.match(line):
            flush_text()
            bullets.append(_inline(_BULLET.sub("", line, count=1)))
        else:
            flush_bullets()
            text.append(_inline(line.strip()))
    flush_text()
    flush_bullets()
    return "".join(out)


def question_html(q):
    return f'<div class="q">{html.escape(q)}</div>'


def sheet_html(text, muted=False):
    body = "".join(_format_block(b) for b in re.split(r"\n\s*\n", text) if b.strip())
    cls = "sheet muted" if muted else "sheet"
    return f'<div class="{cls}">{body}</div>'


def render_sources(hits):
    """Sources stay hidden behind one button."""
    grouped = {}
    for h in hits:
        m = h["meta"]
        grouped.setdefault((m["source"], m["unit"], m["page"]), []).append((h["text"], h["distance"]))
    with st.popover(f"Sources ({len(grouped)})"):
        for (src, unit, page), texts in grouped.items():
            st.markdown(
                f'<span class="mark">{html.escape(src)}</span> &nbsp;{unit} {page}',
                unsafe_allow_html=True,
            )
            for passage, dist in texts:
                st.caption(f"Match {round((1 - dist) * 100)}%: {passage}")


def give_feedback(index, verdict):
    m = st.session_state.messages[index]
    m["feedback"] = verdict
    sources = [
        {"source": h["meta"]["source"], "page": h["meta"]["page"]} for h in (m.get("hits") or [])
    ]
    log_feedback(verdict, m.get("question"), m["content"], sources, refused=not m.get("hits"))


def render_actions(index, m):
    """Sources button plus thumbs up/down under an answer."""
    if m.get("error"):
        return
    c1, c2, c3, c4 = st.columns([1.7, 0.6, 0.6, 5], vertical_alignment="center")
    with c1:
        if m.get("hits"):
            render_sources(m["hits"])
    done = m.get("feedback")
    c2.button("👍", key=f"up_{index}", help="Good answer", disabled=bool(done),
              on_click=give_feedback, args=(index, "up"))
    c3.button("👎", key=f"down_{index}", help="Wrong or unhelpful", disabled=bool(done),
              on_click=give_feedback, args=(index, "down"))
    if done:
        c4.caption("Thanks. Saved in your local feedback log.")


# ---------- Start-up checks ----------
@st.cache_data(ttl=5, show_spinner=False)
def cached_checks():
    return run_checks()


problems = cached_checks()
if problems:
    st.markdown('<h1 class="app-title">Private Document AI</h1><div class="rule"></div>', unsafe_allow_html=True)
    st.write("")
    for p in problems:
        fix = html.escape(p["fix"]).replace("\n", "<br>")
        st.markdown(
            f'<div class="sheet"><p><strong>{html.escape(p["title"])}</strong></p><p>{fix}</p></div>',
            unsafe_allow_html=True,
        )
    if st.button("Check again"):
        cached_checks.clear()
        st.rerun()
    st.stop()

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
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

# ---------- Conversation ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

hint = st.empty()
if not st.session_state.messages:
    msg = (
        "Ask a question. The answer is written only from your documents, and you can open the sources behind it."
        if sources_now
        else "No documents yet. Open Documents to add a PDF, Word or text file."
    )
    hint.markdown(f'<p class="empty">{msg}</p>', unsafe_allow_html=True)

for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        st.markdown(question_html(m["content"]), unsafe_allow_html=True)
    else:
        st.markdown(sheet_html(m["content"]), unsafe_allow_html=True)
        render_actions(i, m)

if question := st.chat_input("Ask a question about your documents"):
    hint.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(question_html(question), unsafe_allow_html=True)

    slot = st.empty()
    slot.markdown(sheet_html("Searching your documents...", muted=True), unsafe_allow_html=True)
    started = time.time()
    trace = []
    failed = False
    try:
        hits, messages, trace = prepare(question)
        if messages is None:
            text, final_hits = REFUSAL, []
        else:
            acc = ""
            for piece in stream_answer(messages):
                acc += piece
                slot.markdown(sheet_html(acc + " ▍"), unsafe_allow_html=True)
            text, final_hits = finalize(acc, hits)
    except Exception:
        failed = True
        text, final_hits = "Could not reach Ollama. Check that it is running, then ask again.", []

    log_query(question, trace, refused=not final_hits, seconds=time.time() - started)
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "hits": final_hits, "question": question, "error": failed}
    )
    st.rerun()  # redraw the conversation so the answer gets its Sources and feedback buttons
