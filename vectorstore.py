import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
META_PATH  = os.path.join(VECTOR_DIR, "metadata.pkl")

def build_vectorstore(chunks):
    """Generate embeddings and store in FAISS index."""

    print("🔄 Generating embeddings...")
    texts     = [chunk["content"]  for chunk in chunks]
    filenames = [chunk["filename"] for chunk in chunks]

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS index built with {index.ntotal} vectors")

    # Save index and metadata
    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({"texts": texts, "filenames": filenames}, f)

    print(f"💾 Saved index  → {INDEX_PATH}")
    print(f"💾 Saved metadata → {META_PATH}")
    return index, texts, filenames


def load_vectorstore(save_path="vectorstore"):
    """Load existing FAISS index from disk."""

    idx_path  = os.path.join(save_path, "index.faiss")
    meta_path = os.path.join(save_path, "metadata.pkl")

    index = faiss.read_index(idx_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Vectorstore loaded with {index.ntotal} vectors")
    return index, metadata["texts"], metadata["filenames"]
