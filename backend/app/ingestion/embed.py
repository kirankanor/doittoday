from uuid import NAMESPACE_URL, uuid5

from ..database.vector_store import VectorStore
from .chunkers import ChunkRecord


def _stable_chunk_id(chunk: ChunkRecord, chunk_index: int) -> str:
    """Create a deterministic ID from the source document and chunk position."""
    source_identity = chunk["source_uri"] or chunk["source_name"]
    if not source_identity:
        raise ValueError(
            "Each chunk needs a source_uri or source_name for stable IDs."
        )
    return str(uuid5(NAMESPACE_URL, f"{source_identity}\0{chunk_index}"))


def embed_and_store(
    chunks: list[ChunkRecord],
    vector_store: VectorStore,
    *,
    collection_name: str,
) -> None:
    """Embed chunk texts and persist them via the shared vector-store API."""
    if not chunks:
        raise ValueError("No text chunks were created; check the source document.")

    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for chunk_index, chunk in enumerate(chunks):
        text = str(chunk["chunk_text"]).strip()
        if not text:
            raise ValueError("Each chunk must include non-empty chunk_text content.")

        texts.append(text)
        ids.append(_stable_chunk_id(chunk, chunk_index))
        metadatas.append(
            {
                "headings": chunk["headings"],
                "content": chunk["content"],
                "source_name": chunk["source_name"],
                "source_uri": chunk["source_uri"],
                "page_numbers": chunk["page_numbers"],
                "chunk_index": chunk_index,
            }
        )

    vector_store.upsert_documents(
        texts=texts,
        ids=ids,
        metadatas=metadatas,
        collection=collection_name,
    )
