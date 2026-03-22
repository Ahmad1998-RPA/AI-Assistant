# app.py
import streamlit as st
import os

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────
st.title("🤖 AI Assistant")
st.markdown("Ask questions from your documents — powered by **llama3 + FAISS**")
st.divider()

# ── Session State ─────────────────────────────────────────
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Auto Pipeline on Startup ──────────────────────────────
if not st.session_state.pipeline_ready:

    # ── Step 1: Ingestion ─────────────────────────────────
    with st.status("🚀 Starting RAG Pipeline...", expanded=True) as status:

        st.write("📄 Step 1: Loading and chunking documents...")
        from ingestion import load_documents, chunk_documents

        if not os.path.exists("documents") or len(os.listdir("documents")) == 0:
            st.error("⚠️ No documents found in /documents folder. Please add files and restart.")
            st.stop()

        docs   = load_documents("documents")
        chunks = chunk_documents(docs)
        st.write(f"✅ Loaded {len(docs)} document(s) → {len(chunks)} chunks")

        # ── Step 2: Vectorstore ───────────────────────────
        st.write("🔄 Step 2: Building vector database...")
        from vectorstore import build_vectorstore
        build_vectorstore(chunks)
        st.write("✅ Vector database built and saved!")

        # ── Step 3: Retriever ─────────────────────────────
        st.write("🔍 Step 3: Initializing retriever...")
        from retriever import retrieve, format_context
        st.write("✅ Retriever ready!")

        # ── Step 4: Generator ─────────────────────────────
        st.write("🤖 Step 4: Loading Gemma:2b via Ollama...")
        from generator import generate_answer
        st.write("✅ Generator ready!")

        status.update(
            label="✅ RAG Pipeline Ready! You can now ask questions.",
            state="complete",
            expanded=False
        )

    st.session_state.pipeline_ready = True
    st.balloons()

else:
    # Already initialized — just import generator
    from generator import generate_answer

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to Retrieve (Top K)", 1, 10, 5)

    st.divider()
    st.subheader("📊 Pipeline Status")
    st.success("✅ Ingestion complete")
    st.success("✅ Vectorstore ready")
    st.success("✅ Retriever ready")
    st.success("✅ Generator ready")

# ── Chat Interface ────────────────────────────────────────
st.subheader("💬 Ask a Question")

if not st.session_state.pipeline_ready:
    st.info("⏳ Pipeline is initializing, please wait...")
else:
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Query input
    query = st.chat_input("Ask something about your documents...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer, sources = generate_answer(query, top_k=top_k)
            st.write(answer)

        st.session_state.chat_history.append({
            "question": query,
            "answer"  : answer,
        })