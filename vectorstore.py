# # vectorstore.py
# import faiss
# import numpy as np
# import pickle
# import os
# from sentence_transformers import SentenceTransformer

# # Load embedding model once
# model = SentenceTransformer("all-MiniLM-L6-v2")


# def build_vectorstore(chunks):
#     """Generate embeddings and store in FAISS index."""
    
#     print("🔄 Generating embeddings...")
#     texts = [chunk["content"] for chunk in chunks]
#     embeddings = model.encode(texts, show_progress_bar=True)
#     embeddings = np.array(embeddings).astype("float32")

#     # Build FAISS index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)

#     print(f"✅ FAISS index built with {index.ntotal} vectors")

#     # Save index and chunks to disk
#     faiss.write_index(index, "vectorstore.index")
#     with open("chunks.pkl", "wb") as f:
#         pickle.dump(chunks, f)

#     print("💾 Vectorstore saved to disk!")
#     return index, chunks


# # def load_vectorstore(save_path="vectorstore"):
# #     """Load existing FAISS index and chunks from disk."""
# #     if not os.path.exists("vectorstore.index") or not os.path.exists("chunks.pkl"):
# #         print("❌ No vectorstore found. Please build it first.")
# #         return None, None

# #     index = faiss.read_index("vectorstore.index")
# #     with open("chunks.pkl", "rb") as f:
# #         chunks = pickle.load(f)

# #     print(f"✅ Vectorstore loaded with {index.ntotal} vectors")
# #     return index, chunks
# def load_vectorstore(save_path="vectorstore"):
#     """Load existing FAISS index from disk."""
    
#     index = faiss.read_index(os.path.join(save_path, "index.faiss"))
    
#     with open(os.path.join(save_path, "metadata.pkl"), "rb") as f:
#         metadata = pickle.load(f)
    
#     print(f"✅ Vectorstore loaded with {index.ntotal} vectors")
#     return index, metadata["texts"], metadata["filenames"]  # ← must return 3 values


# # Test it directly
# if __name__ == "__main__":
#     from ingestion import load_documents, chunk_documents

#     docs = load_documents("documents")
#     chunks = chunk_documents(docs)
#     index, chunks = build_vectorstore(chunks)
#     print("\n✅ Vectorstore build complete!")

# vectorstore.py
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


# # Test it directly
# if __name__ == "__main__":
#     from ingestion import load_documents, chunk_documents

#     docs   = load_documents("documents")
#     chunks = chunk_documents(docs)
#     index, texts, filenames = build_vectorstore(chunks)

#     print("\n--- Vectorstore Test Complete ---")
#     print(f"Total vectors stored: {index.ntotal}")