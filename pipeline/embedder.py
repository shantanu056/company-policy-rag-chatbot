from typing import List
from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document


class TfidfEmbedder:
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = None

    def fit_transform(self, documents: List[Document]):
        texts = [d.page_content for d in documents]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
        )
        vectors = self.vectorizer.fit_transform(texts)
        self._save()
        return vectors

    def transform_query(self, query: str):
        if self.vectorizer is None:
            self._load()
        return self.vectorizer.transform([query])

    def _save(self):
        with open(self.persist_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    def _load(self):
        with open(self.persist_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)


if __name__ == "__main__":
    from pipeline.document_loader import PolicyDocumentLoader
    from pipeline.chunker import PolicyTextChunker

    loader = PolicyDocumentLoader("data/policies")
    docs = loader.load()

    chunker = PolicyTextChunker()
    chunks = chunker.split(docs)

    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform(chunks)

    print(f"Chunks: {len(chunks)}")
    print(f"Embedding matrix shape: {vectors.shape}")
