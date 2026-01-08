#  Company Policy RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about internal company policies using document retrieval and Large Language Models (LLMs).

This project demonstrates a **production-style RAG architecture** with:

* Clean separation of ingestion and inference
* Multiple answer-generation strategies
* Safe handling of sensitive documents
* API-based LLM inference (no local GPU / PyTorch required)

---

##  Key Features

*  **Policy-aware Q&A** using vector search (TF-IDF + cosine similarity)
*  **Deterministic retrieval layer** for grounded responses
*  **Groq + LLaMA 3.1 integration** for concise answer synthesis
*  **Rule-based fallback** when LLMs are unavailable
*  Clean Git hygiene (no raw documents committed)
*  Works reliably on Windows (no CUDA / DLL issues)

---

##  Architecture Overview

```
User Question
     │
     ▼
Retriever (TF-IDF + Cosine Similarity)
     │
     ▼
Top-K Relevant Policy Chunks
     │
     ├── RuleBasedRAG (fallback)
     │
     └── GroqRAGChain (LLaMA 3.1 via Groq API)
     │
     ▼
Final Answer
```

---

##  Project Structure

```
company-policy-rag-chatbot/
│
├── app.py                      # Streamlit UI
├── pipeline/
│   ├── document_loader.py      # PDF ingestion (one-time)
│   ├── chunker.py              # Text chunking
│   ├── embedder.py             # TF-IDF embeddings
│   ├── vector_db.py            # Vector store persistence
│   ├── retriever.py            # Similarity-based retrieval
│   ├── rag_chain.py            # Rule-based RAG (fallback)
│   ├── groq_llm.py             # Groq API wrapper
│   ├── groq_rag_chain.py       # Groq-based RAG
│   ├── hf_llm.py               # Experimental local HF attempt
│   └── hf_rag_chain.py         # Experimental HF RAG (unused)
│
├── data/
│   └── vector_store/           # Prebuilt embeddings (runtime)
│
├── .env                        # API keys (ignored)
├── .gitignore
└── README.md
```

---

##  Why PDFs Are Not in the Repository

Raw policy PDFs are used **only during the ingestion phase** and are intentionally excluded from Git tracking.

Reasons:

* Avoid committing sensitive or proprietary documents
* Prevent Windows file-lock issues
* Follow production best practices

The application runs entirely on a **prebuilt vector store** at runtime.

---

##  LLM Strategy

### Primary (Recommended)

* **Groq API + LLaMA 3.1**
* Fast, high-quality answers
* No local PyTorch or GPU required

### Fallback

* **Rule-based RAG**
* Deterministic, no API dependency

### Experimental

* Local Hugging Face models were implemented but are not used due to
  Windows DLL / PyTorch compatibility issues.
  These files are retained for reference.

---

##  Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\\Scripts\\activate    # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

##  Run the Application

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

##  Example Questions

* *Do we pay for routine eye examinations?*
* *What medical expenses are excluded?*
* *Is hospitalization covered under this policy?*

If the answer is not present in the policy, the chatbot will respond:

> “I don’t know based on the policy documents.”

---

##  Design Philosophy

* Retrieval determines **what information is relevant**
* LLM determines **how the answer is phrased**
* Prompts explicitly restrict hallucination
* Components are swappable and isolated

---

##  Future Improvements

* UI toggle between Rule-based and LLM-based RAG
* Replace TF-IDF with dense embeddings
* Add reranking for improved retrieval
* Support multiple document collections

---
