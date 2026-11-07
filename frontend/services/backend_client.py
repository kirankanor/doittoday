"""HTTP client connecting Streamlit to FastAPI."""
import os
import requests


class BackendError(Exception):
    """Raised when the backend request fails or returns an invalid response."""


def get_backend_url() -> str:
    """Read API base URL from RAG_API_URL."""
    return os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")


def ask_backend(question: str) -> dict:
    """POST question to /chat. Expected response: {answer, citations}.

    If your API requires document_id, add it to the payload below.
    For one internal knowledge collection, backend can resolve it itself.
    """
    try:
        response = requests.post(
            f"{get_backend_url()}/chat",
            json={"question": question, "top_k": 5},
            timeout=240,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise BackendError(str(exc)) from exc
    except ValueError as exc:
        raise BackendError("Backend returned invalid JSON.") from exc

    if not isinstance(data, dict) or "answer" not in data:
        raise BackendError("Expected JSON containing an 'answer' field.")
    return data
