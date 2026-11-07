import re

def _pack(text, spans):
    out = []
    for start, end in spans:
        value = text[start:end].strip()
        if value:
            out.append({"chunk_index": len(out), "text": value, "start": start, "end": end})
    return out

def _validate(size, overlap):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")

def fixed_chunks(text, size=900, overlap=100):
    _validate(size, overlap)
    spans, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return _pack(text, spans)

def recursive_chunks(text, size=900, overlap=100):
    _validate(size, overlap)
    separators = ["\n\n", "\n", ". ", "; ", " "]
    spans, start = [], 0
    while start < len(text):
        hard_end = min(start + size, len(text))
        if hard_end == len(text):
            spans.append((start, hard_end))
            break
        window = text[start:hard_end]
        cut = None
        for sep in separators:
            pos = window.rfind(sep)
            if pos >= size * 0.55:
                cut = start + pos + len(sep)
                break
        end = cut if cut and cut > start else hard_end
        spans.append((start, end))
        start = max(start + 1, end - overlap)
    return _pack(text, spans)

def structure_aware_chunks(text, size=1200, overlap=100):
    _validate(size, overlap)
    heading = re.compile(
        r"(?m)^(?=(?:SECTION\s+\d+|Section\s+\d+|ARTICLE\s+[IVXLC\d]+|"
        r"Article\s+[IVXLC\d]+|CHAPTER\s+[IVXLC\d]+|Chapter\s+[IVXLC\d]+|"
        r"\d+(?:\.\d+)*[.)]?\s+[A-Z]))"
    )
    starts = sorted(set([0] + [m.start() for m in heading.finditer(text) if m.start()]))
    spans = []
    for i, start in enumerate(starts):
        end = starts[i+1] if i+1 < len(starts) else len(text)
        unit = text[start:end]
        if len(unit) <= size:
            spans.append((start, end))
        else:
            spans.extend((start+c["start"], start+c["end"])
                         for c in recursive_chunks(unit, size, overlap))
    return _pack(text, spans)

from collections.abc import Iterable
from typing import Literal, TypedDict

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.base import BaseChunk, BaseChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    TripletTableSerializer,
)
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.page_chunker import PageChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.html import HTMLDocSerializer
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc import DoclingDocument


class ChunkRecord(TypedDict):
    headings: list[str]
    content: str
    chunk_text: str
    source_name: str | None
    source_uri: str | None
    page_numbers: list[int]


class AdvancedChunkingResult(TypedDict):
    serialized_document: str
    serialization_format: Literal["markdown", "html"]
    chunks: list[ChunkRecord]


def _to_chunk_records(
    chunks: Iterable[BaseChunk],
    chunker: BaseChunker,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []

    for chunk in chunks:
        headings = list(chunk.meta.headings or [])
        content = chunk.text.strip()
        origin = chunk.meta.origin
        page_numbers = sorted(
            {
                provenance.page_no
                for item in chunk.meta.doc_items
                for provenance in item.prov
            }
        )
        records.append(
            {
                "headings": headings,
                "content": content,
                "chunk_text": chunker.contextualize(chunk).strip(),
                "source_name": origin.filename if origin else None,
                "source_uri": str(origin.uri) if origin and origin.uri else None,
                "page_numbers": page_numbers,
            }
        )

    return records


def _document_location(
    document: DoclingDocument,
) -> tuple[str | None, str | None, list[int]]:
    origin = document.origin
    page_numbers = sorted(
        {
            provenance.page_no
            for item, _ in document.iterate_items()
            for provenance in item.prov
        }
    )
    return (
        origin.filename if origin else None,
        str(origin.uri) if origin and origin.uri else None,
        page_numbers,
    )


def hybrid_chunking(
    document: DoclingDocument,
    *,
    tokenizer_model: str | None = None,
    max_tokens: int | None = None,
) -> list[ChunkRecord]:
    """Create token-aware chunks, optionally aligned to an embedding tokenizer."""
    if tokenizer_model is None and max_tokens is not None:
        raise ValueError("max_tokens requires tokenizer_model.")

    tokenizer = (
        HuggingFaceTokenizer.from_pretrained(tokenizer_model, max_tokens=max_tokens)
        if tokenizer_model
        else None
    )
    chunker = HybridChunker(tokenizer=tokenizer) if tokenizer else HybridChunker()
    return _to_chunk_records(chunker.chunk(document), chunker)


def hierarchical_chunking(document: DoclingDocument) -> list[ChunkRecord]:
    """Create chunks aligned to Docling's detected document elements."""
    chunker = HierarchicalChunker()
    return _to_chunk_records(chunker.chunk(document), chunker)


def page_based_chunking(document: DoclingDocument) -> list[ChunkRecord]:
    """Create one Docling chunk per page."""
    chunker = PageChunker()
    return _to_chunk_records(chunker.chunk(document), chunker)


def line_based_token_chunking(
    document: DoclingDocument,
    *,
    prefix: str = "",
) -> list[ChunkRecord]:
    """Create token-aware chunks that preserve line boundaries where possible."""
    chunker = LineBasedTokenChunker(prefix=prefix)
    return _to_chunk_records(chunker.chunk(document), chunker)


def advanced_chunking_and_serialization(
    document: DoclingDocument,
    *,
    output_format: Literal["markdown", "html"] = "markdown",
    image_placeholder: str = "<!-- image -->",
    triplet_tables: bool = False,
    tokenizer_model: str | None = None,
    max_tokens: int | None = None,
) -> AdvancedChunkingResult:
    """Return hybrid chunks and a configurable Docling serialization."""
    if output_format == "html":
        if triplet_tables:
            raise ValueError("triplet_tables is only supported with Markdown output.")
        serialized_document = HTMLDocSerializer(doc=document).serialize().text
    elif output_format == "markdown":
        if triplet_tables:
            serializer = MarkdownDocSerializer(
                doc=document,
                table_serializer=TripletTableSerializer(),
                params=MarkdownParams(image_placeholder=image_placeholder),
            )
        else:
            serializer = MarkdownDocSerializer(
                doc=document,
                params=MarkdownParams(image_placeholder=image_placeholder),
            )
        serialized_document = serializer.serialize().text
    else:
        raise ValueError(f"Unsupported serialization format: {output_format}")

    return {
        "serialized_document": serialized_document,
        "serialization_format": output_format,
        "chunks": hybrid_chunking(
            document,
            tokenizer_model=tokenizer_model,
            max_tokens=max_tokens,
        ),
    }


def trivial_chunking(document: DoclingDocument) -> list[ChunkRecord]:
    """Serialize the whole document as one unsplit chunk."""
    content = MarkdownDocSerializer(doc=document).serialize().text.strip()
    if not content:
        return []

    source_name, source_uri, page_numbers = _document_location(document)
    return [
        {
            "headings": [],
            "content": content,
            "chunk_text": content,
            "source_name": source_name,
            "source_uri": source_uri,
            "page_numbers": page_numbers,
        }
    ]

