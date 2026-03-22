# # generator.py
# import ollama
# from retriever import retrieve, format_context

# def generate_answer(query, top_k=3):
#     """Retrieve relevant chunks and generate an answer using Llama3."""
    
#     # Step 1 - Retrieve relevant chunks
#     results = retrieve(query, top_k=top_k)
    
#     if not results:
#         return "❌ No relevant information found in the documents.", []
    
#     # Step 2 - Format context from retrieved chunks
#     context = format_context(results)
    
#     # Step 3 - Build the RAG prompt
#     prompt = f"""You are a helpful assistant. You will be given context extracted from documents.
# Answer the question directly and concisely using the information in the context.
# Do NOT say "the context does not state" or add disclaimers.
# Just answer naturally as if you know the information.
# If truly nothing relevant exists in the context, only then say "I could not find this information in the documents."

# Context:
# {context}

# Question: {query}

# Answer:"""
    
#     # Step 4 - Generate answer using Llama3 via Ollama
#     print("🤖 Generating answer with gemma:2b...")
#     response = ollama.chat(
#         model= "gemma:2b",#"llama3",
#         messages=[{"role": "user", "content": prompt}]
#     )
    
#     answer = response["message"]["content"]
#     return answer, results


# def display_answer(query, top_k=5):
#     """Display the answer with source references."""
    
#     answer, results = generate_answer(query, top_k=top_k)
    
#     print("\n" + "="*60)
#     print(f"❓ Question: {query}")
#     print("="*60)
#     print(f"💬 Answer:\n{answer}")
#     # print("\n--- 📚 Sources Used ---")
#     # for r in results:
#     #     print(f"  Rank {r['rank']} | {r['filename']} | Distance: {r['distance']}")
#     #     print(f"  Preview: {r['content'][:150]}...")
#     #     print()


# # Test it directly
# if __name__ == "__main__":
#     query = "what is the job title of Ahmad Mubarak?"
#     display_answer(query)

# generator.py
import ollama
from retriever import retrieve, format_context

def generate_answer(query, top_k=5):
    """Retrieve relevant chunks and generate an answer using Gemma:2b."""

    # Step 1 - Retrieve relevant chunks
    results = retrieve(query, top_k=top_k)

    if not results:
        return "❌ No relevant information found in the documents.", []

    # Step 2 - Format context from retrieved chunks
    context = format_context(results)

    # Step 3 - Build the RAG prompt
    prompt = f"""Use the context below to answer the question.
Give a short, direct answer only.
Do not add any explanation, disclaimer, or extra sentences after the answer.

Context:
{context}

Question: {query}

Answer (one or two sentences only):"""

    # Step 4 - Generate answer using Gemma:2b via Ollama
    print("🤖 Generating answer with llama3...")
    response = ollama.chat(
        model="llama3", # "llama3", # ← switch to llama3 if you have it set up "gemma:2b"
        messages=[
            {
                "role": "system",
                "content": "You are a factual assistant. Answer only from the context. Never add disclaimers or extra commentary. Stop after answering."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Step 5 - Clean the answer
    answer = response["message"]["content"].strip()

    # Remove any disclaimer sentences if model still adds them
    lines = answer.split("\n")
    clean_lines = [
        line for line in lines
        if not any(phrase in line.lower() for phrase in [
            "i could not find",
            "context does not",
            "cannot answer",
            "does not provide",
            "does not explicitly",
            "not mentioned in",
            "no information"
        ])
    ]
    answer = "\n".join(clean_lines).strip()

    return answer, results


def display_answer(query, top_k=5):
    """Display the answer with source references."""

    answer, results = generate_answer(query, top_k=top_k)

    print("\n" + "="*60)
    print(f"❓ Question: {query}")
    print("="*60)
    print(f"💬 Answer:\n{answer}")
    print("\n--- 📚 Sources Used ---")
    for r in results:
        print(f"  Rank {r['rank']} | {r['filename']} | Distance: {r['distance']}")
        print(f"  Preview: {r['content'][:150]}...")
        print()


# # Test it directly
# if __name__ == "__main__":
#     query = "What is the salary of Ahmad Mubarak?"
#     display_answer(query)