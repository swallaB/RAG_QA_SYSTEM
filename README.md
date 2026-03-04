# 📄 Swiggy Annual Report RAG QA System
### Retrieval-Augmented Generation (RAG) on a Real-World Business Document

This project implements a **Retrieval-Augmented Generation (RAG)** based question-answering system built on the **Swiggy Annual Report (FY 2023–24)**.

Users can ask natural language questions related to the report, and the system retrieves relevant sections and generates **accurate answers grounded strictly in the document**.

To reduce hallucinations, the system uses:
- **Semantic retrieval from a FAISS vector database**
- **Similarity threshold filtering**
- **Strict prompt grounding**

---

# Dataset

**Document:** Swiggy Annual Report (FY 2023–24)  
**Format:** PDF  

Source:  
https://www.swiggy.com/investors

---

# Architecture


# Tech Stack

- Python
- LangChain
- FAISS (Vector Database)
- HuggingFace Embeddings (`BAAI/bge-base-en-v1.5`)
- Google Gemini API
- PyPDF

---


# Quick Start

Clone the repository:
git clone <repo-url>
cd swiggy-rag
Create a virtual environment:


python -m venv venv


Activate environment:

Windows

venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt
Add your Gemini API key in a `.env` file:


GEMINI_API_KEY=your_api_key_here


---

# Run the System

Step 1: Process the document and build the vector database


python src/ingest.py


Step 2: Start the question answering system


python src/rag.py

# Limitations

- Image-based pages in the report cannot be parsed
- CLI interface only (no web UI)

---

# Future Improvements

- OCR support for scanned pages
- Web interface (Streamlit / Gradio)
- Hybrid retrieval methods