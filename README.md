# 🤖 AI Assistant — Local AI Document Q&A

A fully local, privacy-first Retrieval-Augmented Generation (RAG) Assistant that answers questions from your own documents using **LLaMA 3 + FAISS** — no API keys, no cloud, no cost.

---

## 📌 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances LLM responses by grounding them in real documents. Instead of relying on the model's training data alone, RAG:

1. **Retrieves** the most relevant chunks from your documents
2. **Passes** them as context to the LLM
3. **Generates** a grounded, accurate answer

This eliminates hallucinations and ensures every answer comes directly from your documents.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        app.py                           │
│                  (Streamlit UI + Pipeline)               │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────┐     ┌──────────────────────────┐
│    ingestion.py      │────▶│      vectorstore.py       │
│  Load & Chunk Docs   │     │  Embeddings + FAISS Index │
└─────────────────────┘     └────────────┬─────────────┘
                                          │
                                          ▼
                             ┌──────────────────────────┐
                             │       retriever.py        │
                             │   Semantic Similarity     │
                             │        Search             │
                             └────────────┬─────────────┘
                                          │
                                          ▼
                             ┌──────────────────────────┐
                             │       generator.py        │
                             │   LLaMA 3 via Ollama      │
                             │   Grounded Answer Gen     │
                             └──────────────────────────┘
```

### Pipeline Flow
```
App Starts
    │
    ├── Step 1 → ingestion.py   → Load PDF/TXT/MD + Chunk documents
    ├── Step 2 → vectorstore.py → Generate embeddings + Build FAISS index
    ├── Step 3 → retriever.py   → Cache vectorstore in memory
    ├── Step 4 → generator.py   → Connect to LLaMA 3 via Ollama
    │
    └── ✅ Chat interface ready — ask questions!
```

---

## 🛠️ Tech Stack

| Component         | Tool                        | Purpose                          |
|-------------------|-----------------------------|----------------------------------|
| LLM               | LLaMA 3 (via Ollama)        | Local answer generation          |
| Embeddings        | Sentence Transformers       | Convert text to vectors          |
| Vector Database   | FAISS                       | Semantic similarity search       |
| Document Parsing  | PyMuPDF                     | PDF, TXT, Markdown ingestion     |
| UI                | Streamlit                   | Chat interface                   |
| Language          | Python                      | Core development                 |

---

## 📁 Project Structure

```
RAG_Assistant/
│
├── documents/          ← Place your PDF / TXT / MD files here
├── vectorstore/        ← Auto-generated FAISS index (created on first run)
│   ├── index.faiss
│   └── metadata.pkl
│
├── app.py              ← Streamlit UI + pipeline orchestration
├── ingestion.py        ← Document loading and chunking
├── vectorstore.py      ← Embedding generation and FAISS storage
├── retriever.py        ← Semantic search with in-memory caching
├── generator.py        ← LLM answer generation with RAG prompt
├── requirements.txt    ← Python dependencies
└── README.md           ← You are here
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Anaconda (recommended)
- [Ollama](https://ollama.com/download) installed on your machine

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Ahmad1998-RPA/AI-Assistant.git
cd RAG_Assistant
```

### Step 2 — Create and Activate Conda Environment
```bash
conda create -n RAG_env python=3.10 -y
conda activate RAG_env
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Pull LLaMA 3 via Ollama
```bash
ollama pull llama3
```

### Step 5 — Add Your Documents
Place your PDF, TXT, or Markdown files inside the `/documents` folder:
```
documents/
├── your_file.pdf
├── notes.txt
└── report.md
```

### Step 6 — Run the App
```bash
streamlit run app.py
```

The app will automatically:
- Load and chunk your documents
- Build the FAISS vector database
- Initialize the retriever
- Connect to LLaMA 3
- Open the chat interface at `http://localhost:8501`

---

## 📦 Requirements

```
sentence-transformers
faiss-cpu
pymupdf
streamlit
ollama
numpy
```

Install all at once:
```bash
pip install sentence-transformers faiss-cpu pymupdf streamlit ollama numpy
```

---

## 💡 How to Use

1. **Add documents** to the `/documents` folder
2. **Run** `streamlit run app.py`
3. Wait for the pipeline to initialize (first run takes ~30 seconds)
4. **Ask questions** in the chat box
5. Get grounded answers with source references

### Example Questions
```
"What is the salary of Ahmad Mubarak?"
"What is the job title mentioned in the document?"
"Summarize the key points from the report."
```

---

## 🔍 Key Features

- ✅ **Fully Local** — No API keys, no internet required after setup
- ✅ **Privacy-First** — Your documents never leave your machine
- ✅ **Multi-format** — Supports PDF, TXT, and Markdown files
- ✅ **Source Highlighting** — Every answer shows which document and chunk it came from
- ✅ **In-Memory Caching** — Vectorstore cached after first load for fast queries
- ✅ **Chat History** — Full conversation history within the session
- ✅ **Zero Cost** — Runs entirely on free, open-source tools

---

## 🧩 Module Details

### `ingestion.py`
- Loads PDF, TXT, and Markdown files from the `/documents` folder
- Splits documents into overlapping chunks (default: 500 chars, 50 overlap)
- Handles errors gracefully — skips empty or unreadable files

### `vectorstore.py`
- Generates embeddings using `all-MiniLM-L6-v2` (384-dimensional vectors)
- Builds a FAISS `IndexFlatL2` index for fast similarity search
- Saves index and metadata to disk for persistence

### `retriever.py`
- Embeds the user query using the same model
- Searches FAISS for top-K nearest neighbors
- Caches the vectorstore in memory for fast repeated queries

### `generator.py`
- Builds a strict RAG prompt with retrieved context
- Calls LLaMA 3 via Ollama with a system + user message
- Post-processes the response to remove disclaimers

### `app.py`
- Runs the full pipeline automatically on startup
- Displays a step-by-step initialization status
- Provides a clean chat interface with source expanders
- Shows pipeline status and document stats in the sidebar

---

## ⚠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `ollama: command not found` | Install Ollama from https://ollama.com/download |
| `model requires more memory` | Use a lighter model: `ollama pull gemma:2b` |
| `No documents found` | Add files to the `/documents` folder |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Slow first response | Normal — LLaMA 3 loads into memory on first query |

---

## 🚀 Future Improvements

- [ ] Support for multiple document collections
- [ ] Query rewriting / expansion
- [ ] Hybrid search (keyword + semantic)
- [ ] Chat export to PDF
- [ ] Web URL ingestion support

---

## 👨‍💻 Author

**Ahmad Mubarak**
Built as part of a RAG Hands-on Assignment

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
