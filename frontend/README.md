# Streamlit Frontend

This frontend is a Streamlit chat interface for asking questions about the
internal document collection. It sends questions to the FastAPI backend and
displays the returned answers and citations. Document ingestion and vector
database access are handled outside the frontend.

## Requirements

- Python 3.12 or newer
- The FastAPI backend running and reachable by the frontend

The Python dependencies are declared in `backend/pyproject.toml`.

## Install and run

From the repository root, install the project dependencies:

```powershell
uv sync --project backend
```

Start the API in one terminal:

```powershell
uv run --project backend uvicorn app.api:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal. It connects to `http://localhost:8000` by
default; set `RAG_API_URL` to use a different API address:

```powershell
$env:RAG_API_URL = "http://localhost:8000"
uv run --project backend streamlit run frontend/app.py
```

Open the local URL printed by Streamlit in your browser. Keep the backend
running in a separate terminal.

## Backend API

The frontend sends `POST /chat` with this JSON body:

```json
{
  "question": "What is the notice period?",
  "top_k": 5
}
```

The backend should return an `answer` and may include a `citations` array:

```json
{
  "answer": "The notice period is ...",
  "citations": [
    {
      "document_name": "policy.pdf",
      "page_number": 4,
      "chunk_id": "chunk_123",
      "text": "Supporting excerpt..."
    }
  ]
}
```

Each citation can include `document_name` (or `source`), `page_number`,
`chunk_id`, and `text`. If the backend requires a `document_id`, update the
request payload in `services/backend_client.py`.

The API reads `PG_VECTOR_URL` from `backend/.env` and searches all pgvector
collections by default. Set `RAG_COLLECTIONS` to a comma-separated list to
limit the search. Collections must contain indexed document chunks. Answer
generation uses Groq when `GROQ_API_KEY` and `GROQ_MODEL` are configured, and
otherwise falls back to Ollama (`OLLAMA_HOST` and `OLLAMA_MODEL`, defaulting to
`phi4-mini:latest`).

## Troubleshooting

- **Connection error:** Confirm the backend is running and `RAG_API_URL` points
  to its base URL (without `/chat`).
- **Invalid response:** Check that `POST /chat` returns JSON with an `answer`
  field.
