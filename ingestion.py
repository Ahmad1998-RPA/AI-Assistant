# ingestion.py
import os
import fitz  # PyMuPDF

def load_documents(folder_path="documents"):
    """Load PDF, TXT, and Markdown files from the documents folder."""
    documents = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append({"filename": filename, "content": text})
            print(f"✅ Loaded PDF: {filename}")

        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"filename": filename, "content": text})
            print(f"✅ Loaded TXT: {filename}")

        elif filename.endswith(".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"filename": filename, "content": text})
            print(f"✅ Loaded Markdown: {filename}")

    print(f"\n📄 Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks manually — no LangChain needed."""
    chunks = []

    for doc in documents:
        text = doc["content"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "filename": doc["filename"],
                "content": chunk
            })
            start += chunk_size - chunk_overlap  # overlap step

    print(f"✂️  Total chunks created: {len(chunks)}")
    return chunks
