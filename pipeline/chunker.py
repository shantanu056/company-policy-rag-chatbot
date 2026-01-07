# pipeline/chunker.py

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PolicyTextChunker:
    """
    Splits policy documents into overlapping semantic chunks.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


if __name__ == "__main__":
    from pipeline.document_loader import PolicyDocumentLoader

    loader = PolicyDocumentLoader("data/policies")
    documents = loader.load()

    chunker = PolicyTextChunker()
    chunks = chunker.split(documents)

    print(f"Original pages: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    # Print one sample chunk
    print("\nSample chunk:\n")
    print(chunks[0].page_content[:500])
