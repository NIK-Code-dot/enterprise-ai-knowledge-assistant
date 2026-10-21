# Enterprise AI Knowledge Assistant 🤖

An AI-powered knowledge assistant that helps users search, retrieve, and understand information from documents using **Retrieval-Augmented Generation (RAG)** and Large Language Models (LLMs).

🔗 **Live Demo:** https://enterprise-ai-knowledge-assistant-1.onrender.com  
📚 **Backend API Docs:** https://enterprise-ai-knowledge-assistant-fq15.onrender.com/docs

## ✨ Features

- 📄 Document ingestion and text processing
- 🧩 Text chunking for efficient retrieval
- 🔎 Semantic search using vector embeddings
- 🧠 AI-powered question answering with RAG
- 💬 Interactive Streamlit user interface
- ⚡ FastAPI backend for API-based interactions
- 📚 Document upload and knowledge retrieval workflow
- 🏗️ Modular and extensible architecture

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **AI Architecture:** Retrieval-Augmented Generation (RAG)
- **Vector Database:** ChromaDB
- **Embeddings:** FastEmbed
- **LLM Integration:** Groq API
- **Deployment:** Render
- **Version Control:** Git and GitHub

## 🏗️ Architecture

```text
          User
           |
           v
    Streamlit Frontend
           |
           v
      FastAPI Backend
           |
           v
    Document Processing
           |
           v
     Text Chunking
           |
           v
    Embedding Generation
           |
           v
       ChromaDB
           |
           v
   Relevant Context Retrieval
           |
           v
       Groq LLM
           |
           v
    AI-Generated Answer
```

## 📂 Project Structure

```text
enterprise-ai-knowledge-assistant/
├── api/                 # FastAPI backend
├── app/                 # Streamlit frontend
├── configs/             # Configuration files
├── data/                # Local document data
├── src/                 # RAG pipeline and core modules
├── tests/               # Tests
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NIK-Code-dot/enterprise-ai-knowledge-assistant.git
cd enterprise-ai-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Configure the required API credentials and model settings using your local environment configuration. Never commit API keys or secrets to GitHub.

### 5. Run the application

Start the FastAPI backend and Streamlit frontend according to the project's configuration.

Refer to the source code and configuration files for the exact startup commands and required environment variables.

## 🔐 Security

- Never commit API keys, passwords, or credentials.
- Keep `.env` and other secret configuration files out of version control.
- Do not upload confidential documents or private business data.
- Configure production secrets through your deployment provider's environment settings.

## 🎯 Project Goal

The goal of this project is to make enterprise knowledge more accessible by combining semantic document retrieval with AI-generated answers, helping users find relevant information efficiently.

## 👨‍💻 Author

**NIK-Code-dot**

GitHub: https://github.com/NIK-Code-dot

---

*Built with Python, RAG, FastAPI, Streamlit, and LLMs.*
