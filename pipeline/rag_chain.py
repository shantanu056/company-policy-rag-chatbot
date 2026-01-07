"""
RAG Answer Generation 
Generates answers strictly from retrieved context without any LLM.
"""

from typing import List
from langchain_core.documents import Document


class RuleBasedRAG:
    def __init__(self, max_chars: int = 1200):
        self.max_chars = max_chars

    def run(self, query: str, documents: List[Document]) -> str:
        if not documents:
            return "I don't know based on the provided policy documents."

        context = "\n\n".join(doc.page_content for doc in documents)
        context = context[: self.max_chars]

        answer = (
            "Based on the company policy documents, here is the relevant information:\n\n"
            + context
        )
        return answer


if __name__ == "__main__":
    from pipeline.vector_db import VectorStore
    from pipeline.embedder import TfidfEmbedder
    from pipeline.document_loader import PolicyDocumentLoader
    from pipeline.chunker import PolicyTextChunker
    from pipeline.retriever import PolicyRetriever

    # Load & build
    loader = PolicyDocumentLoader("data/policies")
    docs = loader.load()
    chunker = PolicyTextChunker()
    chunks = chunker.split(docs)

    embedder = TfidfEmbedder()
    embeddings = embedder.embed_documents(chunks)

    store = VectorStore()
    store.build(embeddings, chunks)

    retriever = PolicyRetriever(store, embedder)
    rag = RuleBasedRAG()

    query = "What is the leave policy?"
    retrieved_docs = retriever.retrieve(query)
    answer = rag.run(query, retrieved_docs)

    print("Answer:\n")
    print(answer)