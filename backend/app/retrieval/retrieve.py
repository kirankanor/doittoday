from collections.abc import Sequence
from functools import lru_cache
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ..database.config import get_settings
from ..database.vector_store import VectorStore
from ..embeddings.embedding_service import EmbeddingService

BACKEND_DIR = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _get_embeddings(model_name: str) -> EmbeddingService:
    return EmbeddingService.from_model_name(model_name)


def _format_table(results: pd.DataFrame) -> str:
    columns = list(results.columns)
    rows = [
        [
            " ".join(str(value).split())
            for value in row
        ]
        for row in results.itertuples(index=False, name=None)
    ]
    widths = [
        max([len(str(column)), *(len(row[index]) for row in rows)])
        for index, column in enumerate(columns)
    ]

    def format_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header = format_row([str(column) for column in columns])
    body = [format_row(row) for row in rows]
    return "\n".join([border, header, border, *body, border])


def retrieve(
    query_text: str,
    *,
    collection_name: str | Sequence[str],
    limit: int = 3,
) -> pd.DataFrame:
    """Search one or more pgvector collections for relevant chunks."""
    load_dotenv(BACKEND_DIR / ".env")
    dsn = os.getenv("PG_VECTOR_URL", "").strip()
    if not dsn:
        raise RuntimeError("PG_VECTOR_URL is not configured in backend/.env.")

    collection_names = (
        [collection_name]
        if isinstance(collection_name, str)
        else list(collection_name)
    )

    settings = get_settings()
    embeddings = _get_embeddings(settings.embeddings.embedding_model)
    store = VectorStore(
        dsn=dsn,
        embeddings=embeddings,
        collection=collection_names[0] if collection_names else "doittoday",
    )
    try:
        if not collection_names:
            collection_names = store.list_collections()
        if not collection_names:
            return pd.DataFrame()

        missing_collections = [
            name for name in collection_names if not store.collection_exists(name)
        ]
        if missing_collections:
            names = ", ".join(missing_collections)
            raise RuntimeError(
                f"pgvector collection(s) not found: {names}. "
                "Set RAG_COLLECTIONS to existing collections or ingest documents."
            )

        results = [
            result
            for name in collection_names
            for result in store.similarity_search(
                query_text,
                k=limit,
                collection=name,
            )
        ]
        results.sort(key=lambda result: result["distance"])
        return pd.DataFrame(results[:limit])
    finally:
        store.close()


def main() -> None:
    results = retrieve(
        "What documents are required?",
        collection_name=["legal_cases", "citizenship_guides"],
    )
    if results.empty:
        print("No matching chunks found.")
    else:
        display_results = results.drop(columns=["embedding"], errors="ignore")
        print(_format_table(display_results))


if __name__ == "__main__":
    main()
