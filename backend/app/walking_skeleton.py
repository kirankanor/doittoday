"""Persist document chunks, then smoke-test retrieval and generation.

From the repository root:
    uv run --directory backend --project . python -m app.walking_skeleton

The check uses the local Ollama service by default. ``--provider groq`` sends
retrieved context to Groq; ``--provider auto`` tries Groq first when configured.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo

from .database.config import get_settings
from .database.vector_store import VectorStore
from .embeddings.embedding_service import EmbeddingService
from .generation.generate import generate_answer_with_provider
from .ingestion.chunkers import hybrid_chunking
from .ingestion.embed import embed_and_store
from .ingestion.parsers import load_document

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = BACKEND_DIR.parent
DEFAULT_DOCUMENT = REPOSITORY_DIR / "docs" / "docling_benchmark.pdf"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse, embed, persist a document, then test RAG."
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=DEFAULT_DOCUMENT,
        help=(
            "PDF or other Docling-supported document path "
            f"(default: {DEFAULT_DOCUMENT})"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "groq", "auto"),
        default="ollama",
        help=(
            "ollama stays local (default); groq uses the remote Groq API; "
            "auto tries Groq first when configured."
        ),
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Persistent pgvector collection (default: RAG_COLLECTION or vector_db).",
    )
    parser.add_argument(
        "--check-answer",
        action="store_true",
        help="Also generate an answer after ingestion (requires Ollama or Groq).",
    )
    return parser.parse_args()


def run(
    document_path: Path,
    provider: str,
    collection_name: str | None = None,
    check_answer: bool = False,
) -> None:
    load_dotenv(BACKEND_DIR / ".env")
    collection_name = collection_name or os.getenv("RAG_COLLECTION", "vector_db")
    document_path = (
        document_path
        if document_path.is_absolute()
        else (REPOSITORY_DIR / document_path).resolve()
    )
    if not document_path.is_file():
        raise FileNotFoundError(f"Input document does not exist: {document_path}")

    configured_dsn = os.getenv("PG_VECTOR_URL", "").strip()
    if not configured_dsn:
        raise RuntimeError(
            f"PG_VECTOR_URL is not set; configure it in {BACKEND_DIR / '.env'}."
        )
    dsn = make_conninfo(configured_dsn, connect_timeout=5)

    store: VectorStore | None = None

    try:
        print("[1/7] Checking PostgreSQL connection", flush=True)
        try:
            with psycopg.connect(dsn) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
        except psycopg.OperationalError as error:
            raise RuntimeError(
                "Cannot connect to PostgreSQL. Start the PostgreSQL/pgvector "
                "service and verify PG_VECTOR_URL in backend/.env."
            ) from error

        print(f"[2/7] Parsing document: {document_path.name}")
        parsed_document = load_document(str(document_path))

        print("[3/7] Creating Docling chunks")
        chunks = hybrid_chunking(parsed_document)
        if not chunks:
            raise RuntimeError("The document produced no chunks.")
        print(f"      Created {len(chunks)} chunks")

        print("[4/7] Loading the configured embedding model")
        settings = get_settings()
        embeddings = EmbeddingService.from_model_name(
            settings.embeddings.embedding_model
        )

        print("[5/7] Connecting to PostgreSQL/pgvector")
        store = VectorStore(
            dsn=dsn,
            embeddings=embeddings,
            collection=collection_name,
        )
        database_status = store.health_check()
        print(
            "      Connected; pgvector: "
            f"{database_status['pgvector_version']}"
        )
        store.create_collection()
        embed_and_store(
            chunks,
            store,
            collection_name=collection_name,
        )
        print(
            f"      Persisted {len(chunks)} chunks in collection '{collection_name}'"
        )

        print("[6/7] Retrieving relevant context")
        query = chunks[0]["content"][:500].strip()
        if not query:
            raise RuntimeError("The first document chunk has no searchable content.")
        results = store.similarity_search(
            query,
            k=3,
            collection=collection_name,
        )
        if not results:
            raise RuntimeError("Vector search returned no results.")
        context = "\n\n".join(
            str(result["content"]).strip()
            for result in results
            if str(result["content"]).strip()
        )
        if not context:
            raise RuntimeError("Retrieved results contained no text.")

        print("\nPersistent ingestion and retrieval check passed.")
        if check_answer:
            print(f"[7/7] Generating an answer with {provider}")
            answer, used_provider = generate_answer_with_provider(
                context,
                "What does the retrieved document say?",
                provider=provider,
            )
            if not answer.strip():
                raise RuntimeError("The answer provider returned an empty answer.")
            print(f"Provider used: {used_provider}")
            print(f"Answer: {answer}")
    finally:
        if store is not None:
            store.close()


def main() -> None:
    args = _arguments()
    run(args.document, args.provider, args.collection, args.check_answer)


if __name__ == "__main__":
    main()
