"""
Retriever abstraction
Wraps the vector store search into a clean retriever interface.
"""

from typing import List
from langchain_core.documents import Document


class PolicyRetriever:
    def __init__(self, vector_store, embedder, top_k: int = 5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        query_vec = self.embedder.transform_query(query)
        docs = self.vector_store.search(query_vec, top_k=self.top_k)
        return docs


if __name__ == "__main__":
    from pipeline.vector_db import VectorStore
    from pipeline.embedder import TfidfEmbedder
    from pipeline.document_loader import PolicyDocumentLoader
    from pipeline.chunker import PolicyTextChunker

    # Build everything (temporary test)
    loader = PolicyDocumentLoader("data/policies")
    docs = loader.load()

    chunker = PolicyTextChunker()
    chunks = chunker.split(docs)

    embedder = TfidfEmbedder()
    embeddings = embedder.embed_documents(chunks)

    store = VectorStore()
    store.build(embeddings, chunks)

    retriever = PolicyRetriever(store, embedder)

    query = "What is the leave policy?"
    results = retriever.retrieve(query)

    print("Retrieved documents:\n")
    for i, doc in enumerate(results, 1):
        print(f"Doc {i}: {doc.page_content[:200]}...")
