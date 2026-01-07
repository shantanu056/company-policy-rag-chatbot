from typing import List
import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.nn = None
        self.documents: List[Document] = []

    def build(self, embeddings, documents: List[Document], n_neighbors: int = 5):
        self.nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
        )
        self.nn.fit(embeddings)
        self.documents = documents

    def save(self):
        with open(self.persist_dir / "index.pkl", "wb") as f:
            pickle.dump(self.nn, f)
        with open(self.persist_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self):
        with open(self.persist_dir / "index.pkl", "rb") as f:
            self.nn = pickle.load(f)
        with open(self.persist_dir / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query_vector, top_k: int = 5) -> List[Document]:
        distances, indices = self.nn.kneighbors(query_vector, n_neighbors=top_k)
        return [self.documents[i] for i in indices[0]]


if __name__ == "__main__":
    from pipeline.document_loader import PolicyDocumentLoader
    from pipeline.chunker import PolicyTextChunker
    from pipeline.embedder import TfidfEmbedder

    # Load & chunk
    loader = PolicyDocumentLoader("data/policies")
    docs = loader.load()
    chunker = PolicyTextChunker()
    chunks = chunker.split(docs)

    # Embed
    embedder = TfidfEmbedder()
    embeddings = embedder.fit_transform(chunks)

    # Build vector store
    store = VectorStore()
    store.build(embeddings, chunks)
    store.save()

    print("Vector store built and saved successfully")

    # Test search
    query = "What is the leave policy?"
    q_vec = embedder.transform_query(query)
    results = store.search(q_vec)

    print("\nTop retrieved chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"Chunk {i}: {doc.page_content[:200]}...")
