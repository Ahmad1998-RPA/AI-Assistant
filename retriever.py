# retriever.py
import numpy as np
from sentence_transformers import SentenceTransformer
from vectorstore import load_vectorstore

# Use the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=3, save_path="vectorstore"):
    """Retrieve top-k most relevant chunks for a given query."""
    
    # Load vectorstore
    index, texts, filenames = load_vectorstore(save_path)
    
    # Embed the query
    print(f"\n🔍 Query: {query}")
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Collect results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # valid result
            results.append({
                "rank": i + 1,
                "filename": filenames[idx],
                "content": texts[idx],
                "distance": round(float(distances[0][i]), 4)
            })
    
    print(f"✅ Retrieved {len(results)} relevant chunks\n")
    return results


def format_context(results):
    """Format retrieved chunks into a single context string for the LLM."""
    context = ""
    for r in results:
        context += f"[Source: {r['filename']}]\n{r['content']}\n\n"
    return context.strip()


# # Test it directly
# if __name__ == "__main__":
#     query = "How much Salary Ahmad Mubarak is getting?"
#     results = retrieve(query, top_k=3)
    
#     print("--- Retrieved Chunks ---")
#     for r in results:
#         print(f"\nRank {r['rank']} | Source: {r['filename']} | Distance: {r['distance']}")
#         print(f"{r['content'][:200]}...")
    
#     print("\n--- Formatted Context ---")
#     print(format_context(results))