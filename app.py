"""
AI Study Assistant — Streamlit UI
Entry point: streamlit run app.py

Structure:
  - Page config & title
  - Session state init
  - Sidebar (upload + status + reset)
  - File ingestion
  - Backend hook (placeholder → real graph when backend is ready)
  - Chat history display
  - Chat input + response flow
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── 1. Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 AI Study Assistant")
st.caption("Multi-Agent AI powered learning system")

# ── 2. Session State Init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # chat history: [{role, content}, ...]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # FAISS store (built after upload)

if "bm25_store" not in st.session_state:
    st.session_state.bm25_store = None   # BM25 store (built after upload)

if "current_file" not in st.session_state:
    st.session_state.current_file = None  # tracks which file is loaded

# ── 3. Backend Hook ────────────────────────────────────────────────────────────
def get_ai_response(query: str, vector_store=None, bm25_store=None) -> str:
    """
    Placeholder — swap this out once the backend (LangGraph) is wired in.

    Real implementation will look like:
        from graph.study_graph import study_graph
        result = study_graph.invoke({
            "query": query,
            "intent": "", "context": "", "explanation": "",
            "quiz": "", "final_output": "",
            "vector_store": vector_store,
            "bm25_store": bm25_store,
        })
        return result["final_output"]
    """
    # --- TEMPORARY PLACEHOLDER ---
    return (
        f"**[Placeholder response]**\n\n"
        f"You asked: *{query}*\n\n"
        f"Document loaded: {'✅ Yes' if vector_store else '❌ No'}\n\n"
        "_Replace `get_ai_response()` in `app.py` with the real LangGraph call once the backend is ready._"
    )

# ── 4. File Ingestion ──────────────────────────────────────────────────────────
def ingest_file(uploaded_file) -> None:
    """
    Save uploaded file to a temp path → build FAISS + BM25 stores.
    Both stores are always built from the same chunks list (never desynced).
    Temp file is deleted immediately after loading.
    """
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # --- BACKEND HOOK: replace these imports & calls when backend is ready ---
        # from memory.vector_store import load_documents_from_file, split_documents, build_vector_store
        # from memory.bm25_store import BM25Store
        # docs = load_documents_from_file(tmp_path)
        # chunks = split_documents(docs)
        # st.session_state.vector_store = build_vector_store(chunks)
        # st.session_state.bm25_store = BM25Store(chunks)

        # Placeholder: just mark as loaded so UI reflects the upload
        st.session_state.vector_store = "LOADED"   # replace with real store
        st.session_state.bm25_store = "LOADED"     # replace with real store

    finally:
        os.remove(tmp_path)  # always clean up disk

    # Key by (name, size) to detect same-name-different-file edge case
    st.session_state.current_file = (uploaded_file.name, uploaded_file.size)

# ── 5. Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Study Materials")

    # Short instructions
    st.markdown(
        """
        **How to use:**
        1. *(Optional)* Upload a PDF or TXT file
        2. Ask any study question in the chat
        3. Use **Clear Session** to start fresh
        """
    )
    st.divider()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload study material",
        type=["pdf", "txt"],
        help="Upload a PDF or plain-text file to enable document-based answers.",
    )

    # Reset button
    if st.button("🗑️ Clear Session / Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.bm25_store = None
        st.session_state.current_file = None
        st.success("Session cleared!")

    st.divider()

    # Document status indicator
    st.markdown("**Document status:**")
    if st.session_state.vector_store is not None:
        st.success("✅ Document loaded")
        if st.session_state.current_file:
            name, _ = st.session_state.current_file
            st.caption(f"📄 {name}")
    else:
        st.warning("📭 No document loaded")
        st.caption("Chat still works using general knowledge.")

# ── 6. File Upload Guard ───────────────────────────────────────────────────────
# Only re-ingest if a new file is uploaded (avoid redundant processing on reruns)
if uploaded_file is not None:
    file_key = (uploaded_file.name, uploaded_file.size)
    if st.session_state.current_file != file_key:
        with st.spinner("Processing document..."):
            ingest_file(uploaded_file)
        st.sidebar.success(f"✅ Loaded: **{uploaded_file.name}**")
else:
    # User removed file — clear stores so RAG won't run stale data
    if st.session_state.current_file is not None:
        st.session_state.vector_store = None
        st.session_state.bm25_store = None
        st.session_state.current_file = None

# ── 7. Chat History Display ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 8. Chat Input & Response ───────────────────────────────────────────────────
if query := st.chat_input("Ask something like 'Teach me photosynthesis and test me'"):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Agents orchestrating..."):
            response = get_ai_response(
                query=query,
                vector_store=st.session_state.vector_store,
                bm25_store=st.session_state.bm25_store,
            )
        st.markdown(response)

    # Persist to history
    st.session_state.messages.append({"role": "assistant", "content": response})
