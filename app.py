"""app.py - Streamlit interface for Private Document AI (localhost only)."""
import streamlit as st

from ingest import DOCS_DIR, SUPPORTED, delete_source, index_file, list_sources
from rag import LLM_MODEL, answer, unload_models, warm_up

st.set_page_config(page_title="Private Document AI", page_icon="🔒", layout="centered")


@st.cache_resource
def _start():
    warm_up()  # background thread; does not block the page
    return True


_start()


@st.cache_data(show_spinner=False)
def cached_sources():
    return list_sources()


@st.dialog("Documents", width="large")
def documents_dialog():
    uploads = st.file_uploader(
        "Upload PDF, DOCX or TXT",
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
                st.success(f"{up.name}: {n} chunks indexed")
            else:
                st.warning(f"{up.name}: no text found (scanned PDF?)")
        cached_sources.clear()

    st.markdown("**Indexed documents**")
    sources = cached_sources()
    if not sources:
        st.info("Nothing indexed yet.")
    for name in sources:
        c1, c2 = st.columns([8, 1])
        c1.write(name)
        if c2.button("✕", key=f"del_{name}", help="Remove from index"):
            delete_source(name)
            cached_sources.clear()
            st.rerun()

    st.divider()
    if st.button("Unload model (free memory)"):
        unload_models()
        st.success("Model unloaded. It reloads on your next question.")


def citation_line(hits):
    labels = []
    for h in hits:
        m = h["meta"]
        label = f"{m['source']} ({m['unit']} {m['page']})"
        if label not in labels:
            labels.append(label)
    return "📎 " + "  ·  ".join(labels)


def show_sources(hits):
    st.caption(citation_line(hits))
    with st.expander("View source passages", expanded=False):
        for h in hits:
            m = h["meta"]
            st.markdown(f"**{m['source']}, {m['unit']} {m['page']}**")
            st.caption(h["text"])


# ---------- Header ----------
left, right = st.columns([4, 1])
with left:
    st.title("🔒 Private Document AI")
    st.caption(f"100% local ({LLM_MODEL}). Answers come only from your documents.")
with right:
    st.write("")
    if st.button("📁 Documents", use_container_width=True):
        documents_dialog()

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            show_sources(msg["hits"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            text, hits = answer(question)
        st.markdown(text)
        if hits:
            show_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "hits": hits}
    )