# Persistent Document Ingestion

From the repository root, ingest the default `docs/docling_benchmark.pdf` into
the persistent `vector_db` pgvector collection and verify retrieval:

```powershell
uv run --directory backend --project . python -m app.walking_skeleton
```

Choose a different source document or collection with `--document` and
`--collection`. The collection can also be set with `RAG_COLLECTION` in
`backend/.env`. Re-running ingestion upserts chunks with stable IDs; it does not
drop the collection. The Streamlit API searches all pgvector collections by
default, or the collections listed in `RAG_COLLECTIONS`. Add `--check-answer`
to also test generation with the configured provider.
