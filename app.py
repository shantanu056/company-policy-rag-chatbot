import streamlit as st
from pathlib import Path

from pipeline.vector_db import VectorStore
from pipeline.embedder import TfidfEmbedder
from pipeline.retriever import PolicyRetriever
from pipeline.rag_chain import RuleBasedRAG
from pipeline.groq_rag_chain import GroqRAGChain


# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Company Policy Chatbot", layout="wide")
st.title("🏢 Company Policy Chatbot")
st.write("Ask questions strictly based on company policy documents.")

VECTOR_STORE_DIR = "data/vector_store"


# ---------------- LOAD VECTOR STORE ----------------
@st.cache_resource
def load_rag_components():
    embedder = TfidfEmbedder()
    store = VectorStore(VECTOR_STORE_DIR)
    store.load()

    retriever = PolicyRetriever(store, embedder)

    # Choose RAG implementation
    # rag = RuleBasedRAG()   # fallback(without LLM)
    rag = GroqRAGChain()     # Groq + LLaMA 3.1

    return retriever, rag


if not Path(VECTOR_STORE_DIR).exists():
    st.error("Vector store not found. Please build the index first.")
    st.stop()

retriever, rag = load_rag_components()

# ---------------- CHAT UI ----------------
query = st.text_input("Enter your question")

if query:
    with st.spinner("Searching policy documents..."):
        docs = retriever.retrieve(query)

    with st.spinner("Generating answer..."):
        answer = rag.run(query, docs)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("📄 Retrieved Policy Sections"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Section {i}**")
            st.write(doc.page_content)
            st.markdown("---")
