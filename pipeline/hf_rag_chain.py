from typing import List
from langchain_core.documents import Document

from pipeline.hf_llm import HFLLM


class HFRAGChain:
    def __init__(self):
        self.llm = HFLLM()

    def run(self, query: str, documents: List[Document]) -> str:
        if not documents:
            return "I don't know based on the provided policy documents."

        context = "\n".join(doc.page_content for doc in documents)
        prompt = f"""
You are a company policy assistant.
Answer the question using ONLY the context below.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{query}

Answer in one short sentence:
"""

        return self.llm.generate(prompt)


if __name__ == "__main__":
    from pipeline.vector_db import VectorStore
    from pipeline.embedder import TfidfEmbedder
    from pipeline.document_loader import PolicyDocumentLoader
    from pipeline.chunker import PolicyTextChunker
    from pipeline.retriever import PolicyRetriever

    loader = PolicyDocumentLoader("data/policies")
    docs = loader.load()
    chunker = PolicyTextChunker()
    chunks = chunker.split(docs)

    embedder = TfidfEmbedder()
    embeddings = embedder.fit_transform(chunks)

    store = VectorStore()
    store.build(embeddings, chunks)

    retriever = PolicyRetriever(store, embedder)
    rag = HFRAGChain()

    query = "Do you pay for routine eye examinations?"
    retrieved_docs = retriever.retrieve(query)
    print(rag.run(query, retrieved_docs))