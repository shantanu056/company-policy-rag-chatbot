from typing import List
from langchain_core.documents import Document
from pipeline.groq_llm import GroqLLM

class GroqRAGChain:
    def __init__(self):
        self.llm = GroqLLM()

    def run(self, query: str, documents: List[Document]) -> str:
        if not documents:
            return "I don't know based on the provided policy documents."

        # Limit context length defensively
        context = "\n".join(doc.page_content for doc in documents)
        context = context[:4000]

        prompt = f"""
You are a company policy assistant.
Answer the question using ONLY the policy context below.
If the answer is not explicitly stated, say "I don't know based on the policy documents."

Policy context:
{context}

Question:
{query}

Provide a clear, direct answer in one or two sentences:
"""

        return self.llm.generate(prompt)


if __name__ == "__main__":
    # Safe demo: attempt to load built vector store and run an example query if available.
    from pipeline.vector_db import VectorStore
    from pipeline.embedder import TfidfEmbedder
    from pipeline.retriever import PolicyRetriever

    store = VectorStore("data/vector_store")
    try:
        store.load()
    except FileNotFoundError:
        print("Vector store not found. Build the index first (python -m pipeline.vector_db).")
        raise SystemExit(1)

    embedder = TfidfEmbedder()
    retriever = PolicyRetriever(store, embedder)
    rag = GroqRAGChain()

    # Example usage (uncomment to run)
    # query = "Example: Does the policy cover routine eye examinations?"
    # docs = retriever.retrieve(query)
    # print(rag.run(query, docs))

    print("GroqRAGChain loaded. Use via streamlit app or import the class in scripts.")
