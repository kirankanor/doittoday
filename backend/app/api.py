"""HTTP API for the Streamlit document chat interface."""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generation.generate import generate_answer
from .retrieval.retrieve import retrieve

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

app = FastAPI(title="Internal Knowledge Assistant API")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    if not os.getenv("PG_VECTOR_URL", "").strip():
        raise HTTPException(
            status_code=503,
            detail="PG_VECTOR_URL is not configured in backend/.env.",
        )

    collections = [
        name.strip()
        for name in os.getenv("RAG_COLLECTIONS", "").split(",")
        if name.strip()
    ]
    try:
        results = retrieve(question, collection_name=collections, limit=request.top_k)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except psycopg.OperationalError as error:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL is unavailable. Check PG_VECTOR_URL and the database service.",
        ) from error
    if results.empty:
        return {
            "answer": "I couldn't find relevant information in the documents.",
            "citations": [],
        }

    citations = []
    context_parts = []
    for row in results.to_dict(orient="records"):
        text = str(row.get("content") or "").strip()
        metadata = row.get("metadata") or {}
        if text:
            context_parts.append(text)
            pages = metadata.get("page_numbers") or []
            citations.append(
                {
                    "document_name": metadata.get("source_name")
                    or metadata.get("source_uri")
                    or "Unknown document",
                    "page_number": pages[0] if pages else None,
                    "chunk_id": row.get("id"),
                    "text": text,
                }
            )

    if not context_parts:
        return {
            "answer": "I couldn't find relevant information in the documents.",
            "citations": [],
        }

    return {
        "answer": generate_answer("\n\n".join(context_parts), question),
        "citations": citations,
    }