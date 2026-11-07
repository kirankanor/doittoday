# Legal RAG Chunking Playground — Experiments

## What's new in v0.4

The app now has two main tabs:

1. **Chunking Playground** — preserves the existing single-run workflow.
2. **Experiments** — configure and run multiple chunking experiments against the same uploaded Markdown document.

### Experiment builder
- Starts with three editable examples:
  - EXP-01: Fixed-size, 1000 characters, overlap 100
  - EXP-02: Fixed-size, 600 characters, overlap 80
  - EXP-03: Recursive, 700 characters, overlap 110
- Add or remove experiments.
- Each experiment can use any existing strategy:
  - Fixed-size
  - Recursive
  - Structure-aware
  - Hybrid (Docling)
  - Hierarchical (Docling)
  - Page-based (Docling)
  - Line-based token (Docling)
  - Advanced (Docling serialization + Hybrid chunking)
  - Trivial (Docling)
- Strategy-specific configuration fields appear automatically.
- Run all configured experiments in sequence.
- Results are retained in the Streamlit session so you can switch between completed runs without rerunning them.

### Results view
The results area has clickable tabs for:
- Original document
- Each completed experiment

Each experiment tab uses the full available page width and displays all its chunks, metadata, configuration, and JSON download. The Advanced strategy also provides a serialized-document download.

## Run on Windows (PowerShell)

```powershell
uv sync
uv run streamlit run app.py
```

## Important notes
- Upload the same parsed `.md` file for all experiments. Changing the uploaded file clears prior experiment results.
- Fixed-size, Recursive and Structure-aware sizes are measured in characters.
- Docling strategies convert the Markdown into a Docling document before chunking.
- Hybrid and Advanced optionally accept a Hugging Face tokenizer model and maximum token count. The maximum token setting is used only when a tokenizer model is provided.
- The existing chunker implementations in `chunkers.py` have been retained.
- Experiment results are stored in Streamlit session state; they are not yet persisted to disk or a database.
- This is a chunking comparison tool, not a statistical evaluation runner. It does not calculate retrieval or answer-quality metrics.


## Ground-truth table preservation metrics

The original Markdown pipe tables are used as ground truth. The evaluator distinguishes:
- Cell preservation anywhere in the generated chunks.
- Row preservation anywhere in the generated chunks.
- Row integrity: all cells of a row remain together in one chunk.
- Header association: a row and its table header remain together in one chunk.
- Complete table integrity: the header and all rows remain together, in order, in one chunk.
- Table fragmentation: number of chunks containing table terms.
- Tables larger than configured target chunk size (where a target size is configured).

These are text-based structural diagnostics. Repeated/common cell values and flattened tables can create ambiguous matches; review table-level details for uncertain cases. A table larger than the configured target is flagged, not automatically treated as a chunking failure.
