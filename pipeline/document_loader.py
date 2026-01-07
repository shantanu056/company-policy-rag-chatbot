# pipeline/document_loader.py

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class PolicyDocumentLoader:
    """
    Loads company policy PDFs and extracts text page-wise.
    """

    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)

    def load(self) -> List[Document]:
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.pdf_dir}")

        documents: List[Document] = []

        for pdf_file in self.pdf_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf_file.name
                documents.append(page)

        return documents


if __name__ == "__main__":
    loader = PolicyDocumentLoader("data/policies")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from policy documents")
