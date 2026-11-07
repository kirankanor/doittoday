"""Generate context-grounded answers for legal RAG queries.

I updated it: now the normal command prints the overall and grouped metric summaries in the terminal, and still saves the full report—including scores for each question—to report.md.

Run it again:
uv run python evaluation\report_results.py

To print the full per-question table in the terminal too:
uv run python evaluation\report_results.py --show-all


"""

import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from groq import APIError
from langchain_groq import ChatGroq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
OLLAMA_REQUEST_TIMEOUT = 180

SYSTEM_PROMPT = (
    "You are a legal information assistant specializing in Indian citizenship law.\n"
    "Answer using ONLY the provided context; do not make up facts or legal conclusions.\n"
    "If the context does not contain enough information, say so clearly.\n"
    "Cite the relevant case, section, or source when the context identifies one."
)


def generate_answer(context: str, query: str) -> str:
    """Generate an answer using only the supplied retrieved context."""
    answer, _ = generate_answer_with_provider(context, query)
    return answer


def generate_answer_with_provider(
    context: str,
    query: str,
    *,
    provider: str = "auto",
) -> tuple[str, str]:
    """Generate a context-grounded answer and return the provider used."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not context.strip():
        raise ValueError("Context cannot be empty.")
    if provider not in {"auto", "groq", "ollama"}:
        raise ValueError("Provider must be 'auto', 'groq', or 'ollama'.")

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    if provider != "ollama":
        if not api_key or not model:
            groq_error = "GROQ_API_KEY or GROQ_MODEL is not configured"
        else:
            try:
                llm = ChatGroq(
                    model=model,
                    api_key=api_key,
                    temperature=0.2,
                )
                response = llm.invoke(
                    [
                        ("system", SYSTEM_PROMPT),
                        ("user", f"Context:\n{context}\n\nQuestion: {query}"),
                    ]
                )
                if not isinstance(response.content, str):
                    raise TypeError("The answer model returned non-text content.")
                if not response.content.strip():
                    raise ValueError("The answer model returned an empty answer.")
                return response.content, "groq"
            except APIError as error:
                if provider == "groq":
                    raise
                groq_error = type(error).__name__

        if provider == "groq":
            raise ValueError(groq_error)
        logging.warning(
            "Groq is unavailable (%s); falling back to Ollama.",
            groq_error,
        )

    return _generate_with_ollama(context, query), "ollama"


def _generate_with_ollama(context: str, query: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "phi4-mini:latest")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"/no_think\n{SYSTEM_PROMPT}"},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2},
    }
    request = Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_REQUEST_TIMEOUT) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {detail}"
        ) from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Ollama is unavailable at {host} or exceeded "
            f"{OLLAMA_REQUEST_TIMEOUT} seconds; start the service and ensure "
            f"model '{model}' is installed."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError("Ollama returned invalid JSON.") from error

    message = result.get("message") if isinstance(result, dict) else None
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned an empty or invalid answer.")
    return answer