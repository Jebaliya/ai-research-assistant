# AI Research Assistant

An interactive Streamlit research assistant that allows authenticated users to upload documents, index them with a hybrid retrieval pipeline, and ask natural language questions over the uploaded content.

## Features

- Secure login using `streamlit_authenticator` and `auth_config.yaml`
- Upload PDF, DOCX, and TXT files
- Automatic document chunking and indexing
- Persistent Chroma vector store in `chroma_db/`
- Hybrid retrieval combining vector search and BM25
- Response generation with the Groq Gemini LLM
- Agentic workflow mode with retrieval step transparency
- Document comparison for side-by-side topic analysis
- Evaluation metrics using ROUGE and relevance scoring
- Optional trace logging through Langfuse
- Docker and Docker Compose support

## Repository Structure

- `app.py` - Streamlit application entry point and UI logic
- `auth_config.yaml` - authentication configuration and sample users
- `.env` - environment variables for API keys and hosted services
- `requirements.txt` - Python dependencies
- `dockerfile` - Docker build definition
- `docker-compose.yml` - local deployment orchestration
- `chroma_db/` - persistent document embedding storage
- `rag/` - retrieval-augmented generation modules
  - `document_loader.py` - parse PDFs, DOCX, and TXT files
  - `chunker.py` - split text into overlapping chunks
  - `vector_store.py` - Chroma persistence and query wrapper
  - `hybrid_search.py` - BM25 retrieval for lexical search
  - `retriever.py` - combines vector and BM25 search results
  - `llm.py` - Groq Gemini language model integration
  - `agent.py` - stepwise research agent workflow
  - `evaluator.py` - question-answer evaluation metrics
  - `logger.py` - optional Langfuse trace logging

## Requirements

- Python 3.10+
- Streamlit
- ChromaDB
- Groq Python client
- PyMuPDF
- python-docx
- python-dotenv
- streamlit-authenticator
- PyYAML
- rank_bm25
- rouge-score
- langfuse

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Rename or copy `.env` if needed, then add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://jp.cloud.langfuse.com
```

2. Update `auth_config.yaml` to add or modify authenticated users.

> The repository includes example credentials for `admin` and `jayveer`.

## Running Locally

Activate your Python environment and run:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Running with Docker

Use Docker Compose to build and run the app:

```bash
docker-compose up --build
```

The service is exposed at `http://localhost:8501`.

## Usage

1. Log in with a configured username and password.
2. Upload PDF, DOCX, or TXT files using the sidebar.
3. Ask questions about uploaded documents using the chat input.
4. Enable `Agentic Workflow` to see agent reasoning steps.
5. Compare two uploaded documents on a topic in the sidebar.
6. Review evaluation metrics after each query.

## Notes

- Uploaded documents are persisted in `chroma_db/` for reuse.
- Missing `GROQ_API_KEY` will cause the app to display an API key error.
- Document comparison reuses the same retrieval and LLM pipeline.
- Langfuse is optional; tracing is skipped if keys are missing.

## Troubleshooting

- If login fails, check `auth_config.yaml` and authentication cookie settings.
- If uploads fail, confirm supported file types: `.pdf`, `.docx`, `.txt`.
- If the LLM fails, verify `GROQ_API_KEY` and network connectivity.
- If Langfuse tracing fails, confirm `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`.

## License

This repository does not include a license file. Add one before sharing or distributing the project.
