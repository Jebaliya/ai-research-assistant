# AI Research Assistant

> A Streamlit-based research assistant that lets authenticated users upload documents, index them with a hybrid retrieval engine, and ask natural language questions against the uploaded content.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ What this project does

- 📄 Loads PDF, DOCX, and TXT documents
- ✂️ Splits text into overlapping chunks
- 🗄️ Stores chunks in a persistent Chroma vector database
- 🔍 Performs hybrid retrieval using Chroma embeddings and BM25
- 🤖 Generates answers with a Groq Gemini LLM
- 🧠 Supports an agentic workflow with step-by-step reasoning
- 📊 Provides document comparison and evaluation scoring
- 📡 Logs queries and uploads to Langfuse when configured

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd "AI Research Assistant"
```

### 2. Create and activate a Python virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the repository root with the following values:

```env
GROQ_API_KEY=your_groq_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://jp.cloud.langfuse.com
```

### 5. Configure authentication

Update `auth_config.yaml` to add or modify your allowed users. The repository includes two example users:

- `admin`
- `jayveer`

Passwords are stored as bcrypt hashes in the file.

### 6. Verify the Chroma database directory

The app uses `chroma_db/` for persistent storage. The directory is created automatically when the app starts.

---

## 🚀 Running the app locally

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🐳 Running with Docker

Build and run the container using Docker Compose:

```bash
docker-compose up --build
```

The service will be available at `http://localhost:8501`.

---

## 🗂️ Application structure

```
.
├── app.py                  # main Streamlit UI and workflow
├── auth_config.yaml        # login credentials and cookie settings
├── .env                    # API and service configuration
├── requirements.txt        # Python dependencies
├── dockerfile              # Docker image definition
├── docker-compose.yml      # local deployment orchestration
├── chroma_db/              # persistent vector store directory
└── rag/                    # retrieval-augmented generation modules
    ├── document_loader.py  # file parsing
    ├── chunker.py          # text chunking
    ├── vector_store.py     # Chroma persistence and query wrapper
    ├── hybrid_search.py    # BM25 lexical search
    ├── retriever.py        # combines retrieval signals
    ├── llm.py              # Groq Gemini integration
    ├── agent.py            # agent workflow logic
    ├── evaluator.py        # evaluation metrics for answers
    └── logger.py           # optional Langfuse tracing
```

---

## 📖 Usage

1. Log in via the Streamlit login form.
2. Upload one or more documents from the sidebar.
3. Ask a question in the chat input field.
4. View answer evaluation metrics in the sidebar.
5. Enable `Agentic Workflow` for retrieval step details.
6. Compare two uploaded documents using the `Compare Documents` panel.

---

## 📝 Notes

- Supported upload formats: `.pdf`, `.docx`, `.txt`
- Document chunks are indexed and stored in `chroma_db/`
- If `GROQ_API_KEY` is missing, the app shows an API key error
- The comparison feature uses the same retrieval and LLM pipeline
- Langfuse tracing is optional and only active when keys are present
