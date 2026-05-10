FEE Regulations AI Assistant
# FEE Regulations AI Assistant

An AI-powered chatbot designed for the Faculty of Electronic Engineering, Menoufia University.  
The system uses Retrieval-Augmented Generation (RAG) with LangChain and ChromaDB to answer students’ questions directly from the official faculty regulations PDF.

---

# Features

- PDF-based Question Answering
- RAG Architecture using LangChain
- Semantic Search with HuggingFace Embeddings
- Fast Inference using Groq + Llama 3.1
- Multilingual Support (Arabic & English)
- Context-Aware Responses
- Page Reference Retrieval

---

# Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API
- Llama 3.1
- PyPDF
- Google Colab

---

# Project Architecture

```text
PDF Documents
      ↓
Text Chunking
      ↓
Embeddings Generation
      ↓
Vector Database (ChromaDB)
      ↓
Retriever
      ↓
LLM (Llama 3.1 via Groq)
      ↓
Answer Generation
```

# Installation

```bash
git clone https://github.com/yourusername/FEE-RAG-Chatbot.git

cd FEE-RAG-Chatbot

pip install -r requirements.txt
```

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

# Usage

```python
response = ask_bot("What are the graduation requirements?")
print(response)
```

# Example

## User Question

```text
What is the maximum number of credit hours allowed?
```

## Assistant Response

```text
According to the faculty regulations...
```

# Future Improvements

- Web Interface with Gradio
- Voice Assistant Integration
- Fine-tuned Local LLM
- Hybrid Search
- Conversation Memory

# Author

Mohamed Salah  
AI Engineer | Computer Vision & LLM Applications
