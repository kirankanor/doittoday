import json
import uuid

import markdown ,re , statistics , math
import streamlit as st
import streamlit.components.v1 as components
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from chunkers import (
    advanced_chunking_and_serialization,
    fixed_chunks,
    hierarchical_chunking,
    hybrid_chunking,
    line_based_token_chunking,
    page_based_chunking,
    recursive_chunks,
    structure_aware_chunks,
    trivial_chunking,
)
from utils import approx_tokens, extract_text


st.set_page_config(page_title="Legal RAG Chunking Playground", layout="wide")

DOCLING_STRATEGIES = {
    "Hybrid": hybrid_chunking,
    "Hierarchical": hierarchical_chunking,
    "Page-based": page_based_chunking,
    "Line-based token": line_based_token_chunking,
    "Trivial": trivial_chunking,
}
ALL_STRATEGIES = [
    "Fixed-size",
    "Recursive",
    "Structure-aware",
    *DOCLING_STRATEGIES.keys(),
    "Advanced",
]


@st.cache_resource
def get_converter():
    return DocumentConverter()


def normalize_chunks(chunks):
    normalized = []
    for index, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", chunk.get("chunk_text", chunk.get("content", "")))
        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {"text", "chunk_text", "content"}
        }
        normalized.append(
            {"chunk_index": index, "text": str(chunk_text), "metadata": metadata}
        )
    return normalized


def render_comparison(text, chunks, title="Chunking result"):
    """Render original and chunk stream in synchronized, independently scrollable panels."""
    original_html = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"]
    )
    chunk_sections = []
    for i, chunk in enumerate(chunks):
        chunk_html = markdown.markdown(
            chunk["text"], extensions=["tables", "fenced_code", "sane_lists"]
        )
        chunk_metadata = {
            "chunk_index": chunk["chunk_index"],
            **chunk["metadata"],
        }
        chunk_sections.append(
            f'<section class="chunk"><div class="chunk-meta">'
            f'Chunk {i + 1} · {len(chunk["text"]):,} characters '
            f'· ~{approx_tokens(chunk["text"]):,} tokens</div>'
            f'<div class="markdown-body">{chunk_html}</div>'
            f'<details><summary>Chunk metadata</summary>'
            f'<pre>{json.dumps(chunk_metadata, ensure_ascii=False, indent=2, default=str)}</pre>'
            f'</details></section>'
        )

    original_json = json.dumps(original_html, ensure_ascii=False).replace("</", "<\\/")
    chunks_json = json.dumps("".join(chunk_sections), ensure_ascii=False).replace("</", "<\\/")
    html_doc = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; color: var(--text, #f5f5f5); font-family: sans-serif; }}
      .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
      .panel-wrap {{ min-width: 0; }}
      .panel-title {{ font-size: 22px; font-weight: 700; margin: 0 0 14px; }}
      .panel {{ height: 700px; overflow-y: auto; overflow-x: hidden;
                border: 1px solid #555; border-radius: 9px; padding: 18px;
                background: var(--secondary-background-color, #111318);
                scrollbar-gutter: stable; }}
      .markdown-body {{ overflow-wrap: anywhere; line-height: 1.65; font-size: 16px; }}
      .markdown-body img {{ max-width: 100%; height: auto; }}
      .markdown-body pre {{ white-space: pre-wrap; overflow-wrap: anywhere;
                            padding: 12px; border-radius: 6px; background: #20232a; }}
      .markdown-body table {{ border-collapse: collapse; width: 100%; }}
      .markdown-body th, .markdown-body td {{ border: 1px solid #777; padding: 6px; }}
      .chunk {{ padding: 2px 0 16px; margin-bottom: 18px; border-bottom: 1px solid #666; }}
      .chunk-meta {{ font-size: 14px; font-weight: 700; color: #b7c4d6; margin-bottom: 14px; }}
      details {{ margin-top: 12px; font-size: 13px; }}
      details pre {{ white-space: pre-wrap; }}
      @media (max-width: 800px) {{
        .panels {{ grid-template-columns: 1fr; }}
        .panel {{ height: 500px; }}
      }}
    </style>
    </head>
    <body>
      <div class="panels">
        <div class="panel-wrap">
          <div class="panel-title">Original parsed Markdown</div>
          <div class="panel" id="original-panel"><div class="markdown-body" id="original-content"></div></div>
        </div>
        <div class="panel-wrap">
          <div class="panel-title">{title} ({len(chunks)} chunks)</div>
          <div class="panel" id="chunks-panel"><div id="chunks-content"></div></div>
        </div>
      </div>
      <script>
        const originalHtml = {original_json};
        const chunksHtml = {chunks_json};
        document.getElementById("original-content").innerHTML = originalHtml;
        document.getElementById("chunks-content").innerHTML = chunksHtml;
        const left = document.getElementById("original-panel");
        const right = document.getElementById("chunks-panel");
        let syncing = false;
        function syncScroll(source, target) {{
          if (syncing) return;
          syncing = true;
          const sourceRange = source.scrollHeight - source.clientHeight;
          const targetRange = target.scrollHeight - target.clientHeight;
          const progress = sourceRange > 0 ? source.scrollTop / sourceRange : 0;
          target.scrollTop = progress * Math.max(0, targetRange);
          requestAnimationFrame(() => {{ syncing = false; }});
        }}
        left.addEventListener("scroll", () => syncScroll(left, right), {{ passive: true }});
        right.addEventListener("scroll", () => syncScroll(right, left), {{ passive: true }});
      </script>
    </body>
    </html>
    """
    components.html(html_doc, height=760, scrolling=False)


def new_experiment(number):
    return {
        "id": uuid.uuid4().hex[:8],
        "name": f"EXP-{number:02d}",
        "strategy": "Fixed-size",
    }


def ensure_experiment_state():
    if "experiment_configs" not in st.session_state:
        defaults = [
            {"id": "exp001", "name": "EXP-01", "strategy": "Fixed-size"},
            {"id": "exp002", "name": "EXP-02", "strategy": "Fixed-size"},
            {"id": "exp003", "name": "EXP-03", "strategy": "Recursive"},
        ]
        st.session_state.experiment_configs = defaults
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = {}


def execute_strategy(strategy, text, filename, config):
    """Run one configured strategy and return normalized chunks plus optional serialization."""
    serialized_document = None

    if strategy == "Fixed-size":
        chunks = fixed_chunks(
            text,
            size=int(config["chunk_size"]),
            overlap=int(config["overlap"]),
        )
    elif strategy == "Recursive":
        chunks = recursive_chunks(
            text,
            size=int(config["chunk_size"]),
            overlap=int(config["overlap"]),
        )
    elif strategy == "Structure-aware":
        chunks = structure_aware_chunks(
            text,
            size=int(config["chunk_size"]),
            overlap=int(config["overlap"]),
        )
    else:
        document = get_converter().convert_string(
            text,
            format=InputFormat.MD,
            name=filename or "document.md",
        ).document

        if strategy == "Hybrid":
            tokenizer_model = config.get("tokenizer_model") or None
            max_tokens = config.get("max_tokens")
            chunks = hybrid_chunking(
                document,
                tokenizer_model=tokenizer_model,
                max_tokens=int(max_tokens) if tokenizer_model and max_tokens else None,
            )
        elif strategy == "Hierarchical":
            chunks = hierarchical_chunking(document)
        elif strategy == "Page-based":
            chunks = page_based_chunking(document)
        elif strategy == "Line-based token":
            chunks = line_based_token_chunking(
                document,
                prefix=config.get("prefix", ""),
            )
        elif strategy == "Trivial":
            chunks = trivial_chunking(document)
        elif strategy == "Advanced":
            tokenizer_model = config.get("tokenizer_model") or None
            max_tokens = config.get("max_tokens")
            result = advanced_chunking_and_serialization(
                document,
                output_format=config.get("serialization_format", "markdown"),
                image_placeholder=config.get("image_placeholder", "<!-- image -->"),
                triplet_tables=bool(config.get("triplet_tables", False)),
                tokenizer_model=tokenizer_model,
                max_tokens=int(max_tokens) if tokenizer_model and max_tokens else None,
            )
            chunks = result["chunks"]
            serialized_document = result["serialized_document"]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    return normalize_chunks(chunks), serialized_document


def render_experiment_config(config, index):
    exp_id = config["id"]
    title_key = f"cfg_{exp_id}_name"
    strategy_key = f"cfg_{exp_id}_strategy"

    with st.container(border=True):
        top_left, top_right = st.columns([5, 1])
        with top_left:
            st.text_input("Experiment name", value=config["name"], key=title_key)
        with top_right:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Remove", key=f"remove_{exp_id}", disabled=len(st.session_state.experiment_configs) <= 1):
                st.session_state.experiment_configs = [
                    item for item in st.session_state.experiment_configs
                    if item["id"] != exp_id
                ]
                st.rerun()

        strategy = st.selectbox(
            "Chunking strategy",
            ALL_STRATEGIES,
            index=ALL_STRATEGIES.index(config["strategy"]),
            key=strategy_key,
        )
        values = {"name": st.session_state[title_key], "strategy": strategy}

        if strategy in {"Fixed-size", "Recursive", "Structure-aware"}:
            col_a, col_b = st.columns(2)
            values["chunk_size"] = col_a.number_input(
                "Chunk size (characters)", min_value=100, max_value=20000,
                value=int(st.session_state.get(f"cfg_{exp_id}_size", 1000)),
                step=100, key=f"cfg_{exp_id}_size",
            )
            values["overlap"] = col_b.number_input(
                "Overlap (characters)", min_value=0, max_value=10000,
                value=int(st.session_state.get(f"cfg_{exp_id}_overlap", 100)),
                step=10, key=f"cfg_{exp_id}_overlap",
            )
            if values["overlap"] >= values["chunk_size"]:
                st.error("Overlap must be smaller than chunk size.")
        elif strategy in {"Hybrid", "Advanced"}:
            st.caption("Docling token-aware strategy. Leave tokenizer model empty to use Docling defaults.")
            values["tokenizer_model"] = st.text_input(
                "Hugging Face tokenizer model (optional)",
                value=st.session_state.get(f"cfg_{exp_id}_tokenizer", ""),
                key=f"cfg_{exp_id}_tokenizer",
                placeholder="e.g. sentence-transformers/all-MiniLM-L6-v2",
            )
            values["max_tokens"] = st.number_input(
                "Maximum tokens (optional; requires tokenizer model)",
                min_value=64, max_value=8192,
                value=int(st.session_state.get(f"cfg_{exp_id}_max_tokens", 512)),
                step=64, key=f"cfg_{exp_id}_max_tokens",
            )
            if strategy == "Advanced":
                values["serialization_format"] = st.selectbox(
                    "Serialized document format",
                    ["markdown", "html"],
                    key=f"cfg_{exp_id}_format",
                )
                triplet_requested = st.checkbox(
                    "Serialize tables as triplets (Markdown only)",
                    value=bool(st.session_state.get(f"cfg_{exp_id}_triplet", False)),
                    key=f"cfg_{exp_id}_triplet",
                    disabled=values["serialization_format"] != "markdown",
                )
                values["triplet_tables"] = (
                    triplet_requested and values["serialization_format"] == "markdown"
                )
                values["image_placeholder"] = st.text_input(
                    "Image placeholder",
                    value=st.session_state.get(f"cfg_{exp_id}_image_placeholder", "<!-- image -->"),
                    key=f"cfg_{exp_id}_image_placeholder",
                )
        elif strategy == "Line-based token":
            values["prefix"] = st.text_input(
                "Chunk prefix (optional)",
                value=st.session_state.get(f"cfg_{exp_id}_prefix", ""),
                key=f"cfg_{exp_id}_prefix",
            )
        else:
            st.caption("This strategy uses its built-in Docling defaults; no extra parameters are exposed.")

        return values



def _percent(value):
    return None if value is None else round(value * 100, 1)


def _nonempty(value):
    return value not in (None, "", [], {})


def _merge_spans(chunks, source_length):
    spans = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        start, end = meta.get("start"), meta.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= source_length:
            spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return spans, merged


def _adjacent_text_overlap(chunks, max_check=2000):
    total_overlap = 0
    for left, right in zip(chunks, chunks[1:]):
        a = re.sub(r"\s+", " ", left["text"]).strip()
        b = re.sub(r"\s+", " ", right["text"]).strip()
        limit = min(len(a), len(b), max_check)
        overlap = 0
        for n in range(limit, 19, -1):
            if a[-n:] == b[:n]:
                overlap = n
                break
        total_overlap += overlap
    total_chars = sum(len(c["text"]) for c in chunks)
    return (total_overlap / total_chars) if total_chars else 0.0



def _normalize_table_cell(value):
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", str(value))
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("&amp;", "&").replace("&nbsp;", " ")
    value = re.sub(r"[*_`~]", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _split_markdown_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def extract_ground_truth_tables(source_text):
    """Extract Markdown pipe tables from the uploaded original Markdown."""
    lines = source_text.splitlines()
    tables = []
    i = 0
    while i < len(lines) - 1:
        header_line = lines[i].strip()
        separator = lines[i + 1].strip()
        is_pipe_header = "|" in header_line and re.match(
            r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", separator
        )
        if not is_pipe_header:
            i += 1
            continue

        headers = _split_markdown_row(header_line)
        rows = []
        j = i + 2
        while j < len(lines):
            line = lines[j].strip()
            if not line or "|" not in line:
                break
            cells = _split_markdown_row(line)
            if cells and any(cell.strip() for cell in cells):
                rows.append(cells)
            j += 1

        tables.append({
            "table_id": f"GT-T{len(tables) + 1:02d}",
            "line_start": i + 1,
            "line_end": j,
            "headers": headers,
            "rows": rows,
        })
        i = max(j, i + 1)
    return tables


def _ordered_terms_present(text, terms):
    """Return True when non-empty normalized terms appear in order in text."""
    cursor = 0
    for term in terms:
        if not term:
            continue
        position = text.find(term, cursor)
        if position < 0:
            return False
        cursor = position + len(term)
    return True


def evaluate_tables_against_ground_truth(source_text, chunks, target_chunk_size=None):
    """
    Compare generated chunks with tables extracted from the original Markdown.

    Distinguishes:
      - cell preservation anywhere in the output,
      - complete rows preserved somewhere,
      - complete rows co-located in one chunk,
      - complete tables (header + every row) co-located in one chunk.
    """
    ground_truth = extract_ground_truth_tables(source_text)
    chunk_texts = [str(c.get("text", "")) for c in chunks]
    normalized_chunks = [_normalize_table_cell(text) for text in chunk_texts]
    combined = "\n".join(normalized_chunks)

    if not ground_truth:
        return {
            "available": False,
            "reason": "No Markdown pipe tables were detected in the original document.",
            "table_count": 0,
            "table_detection_recall": None,
            "header_preservation": None,
            "row_preservation": None,
            "cell_preservation": None,
            "row_integrity": None,
            "complete_table_integrity": None,
            "header_association": None,
            "fragmentation": None,
            "oversized_table_count": None,
            "details": [],
        }

    details = []
    total_headers = total_headers_found = 0
    total_rows = total_rows_found = 0
    total_cells = total_cells_found = 0
    total_rows_intact = 0
    total_rows_with_header = 0
    detected_tables = 0
    complete_tables = 0
    fragment_counts = []
    oversized_table_count = 0
    known_size_tables = 0

    for table in ground_truth:
        headers = [_normalize_table_cell(x) for x in table["headers"]]
        headers = [x for x in headers if x]
        rows = [
            [_normalize_table_cell(x) for x in row]
            for row in table["rows"]
        ]
        rows = [[cell for cell in row if cell] for row in rows]
        rows = [row for row in rows if row]

        all_cells = headers + [cell for row in rows for cell in row]
        header_hits = [h for h in headers if any(h in chunk for chunk in normalized_chunks)]
        cell_hits = [cell for cell in all_cells if cell in combined]

        row_hits = 0
        row_intact = 0
        rows_with_header = 0

        for row in rows:
            # Content preservation: every cell can exist anywhere in the output.
            if row and all(cell in combined for cell in row):
                row_hits += 1

            # Row integrity: every cell of a row appears, in order, in one chunk.
            row_in_chunk = [
                idx for idx, chunk in enumerate(normalized_chunks)
                if _ordered_terms_present(chunk, row)
            ]
            if row_in_chunk:
                row_intact += 1

                # Header association: a complete row and the table header appear
                # together in at least one chunk. This is intentionally strict.
                if any(
                    _ordered_terms_present(normalized_chunks[idx], headers)
                    and _ordered_terms_present(normalized_chunks[idx], row)
                    for idx in row_in_chunk
                ):
                    rows_with_header += 1

        # A complete table requires all headers and all rows in the same chunk,
        # preserving their sequence. This is stricter than row integrity.
        complete_table_chunk = None
        ordered_table_terms = headers + [cell for row in rows for cell in row]
        for idx, chunk in enumerate(normalized_chunks):
            if ordered_table_terms and _ordered_terms_present(chunk, ordered_table_terms):
                complete_table_chunk = idx
                break
        table_complete = complete_table_chunk is not None
        if table_complete:
            complete_tables += 1

        table_detected = bool(header_hits or row_hits)
        if table_detected:
            detected_tables += 1

        # Count chunks containing at least one distinctive table cell/header.
        # Common values can overestimate fragmentation; this remains diagnostic.
        table_terms = [term for term in all_cells if len(term) >= 3]
        matching_chunk_indices = [
            idx for idx, chunk in enumerate(normalized_chunks)
            if any(term in chunk for term in table_terms)
        ]
        fragment_counts.append(len(matching_chunk_indices))

        table_source_text = "\n".join(
            [" | ".join(table["headers"])] + [" | ".join(row) for row in table["rows"]]
        )
        table_chars = len(table_source_text)
        if target_chunk_size is not None:
            known_size_tables += 1
            if table_chars > target_chunk_size:
                oversized_table_count += 1

        total_headers += len(headers)
        total_headers_found += len(header_hits)
        total_rows += len(rows)
        total_rows_found += row_hits
        total_cells += len(all_cells)
        total_cells_found += len(cell_hits)
        total_rows_intact += row_intact
        total_rows_with_header += rows_with_header

        details.append({
            "Table": table["table_id"],
            "Source lines": f'{table["line_start"]}-{table["line_end"]}',
            "Rows": len(rows),
            "Approx. table chars": table_chars,
            "Headers preserved": f"{len(header_hits)}/{len(headers)}",
            "Rows preserved": f"{row_hits}/{len(rows)}",
            "Rows intact in one chunk": f"{row_intact}/{len(rows)}",
            "Rows with header in same chunk": f"{rows_with_header}/{len(rows)}",
            "Complete table in one chunk": "Yes" if table_complete else "No",
            "Chunks touched": len(matching_chunk_indices),
            "Larger than target size": (
                "Yes" if target_chunk_size is not None and table_chars > target_chunk_size
                else ("No" if target_chunk_size is not None else "N/A")
            ),
        })

    return {
        "available": True,
        "reason": "",
        "table_count": len(ground_truth),
        "table_detection_recall": detected_tables / len(ground_truth),
        "header_preservation": total_headers_found / total_headers if total_headers else None,
        "row_preservation": total_rows_found / total_rows if total_rows else None,
        "cell_preservation": total_cells_found / total_cells if total_cells else None,
        "row_integrity": total_rows_intact / total_rows if total_rows else None,
        "complete_table_integrity": complete_tables / len(ground_truth),
        "header_association": total_rows_with_header / total_rows if total_rows else None,
        "fragmentation": statistics.mean(fragment_counts) if fragment_counts else None,
        "oversized_table_count": (
            oversized_table_count if known_size_tables else None
        ),
        "details": details,
    }

def calculate_chunk_metrics(source_text, chunks, target_chunk_size=None):
    """Calculate intrinsic, document-based chunk metrics; no LLM calls are made."""
    texts = [c.get("text", "") for c in chunks]
    nonempty_texts = [t for t in texts if t.strip()]
    sizes_chars = [len(t) for t in nonempty_texts]
    sizes_tokens = [approx_tokens(t) for t in nonempty_texts]

    count = len(chunks)
    total_chunk_chars = sum(sizes_chars)
    mean_chars = statistics.mean(sizes_chars) if sizes_chars else 0
    median_tokens = statistics.median(sizes_tokens) if sizes_tokens else 0
    p95_tokens = (
        sorted(sizes_tokens)[min(len(sizes_tokens) - 1, math.ceil(0.95 * len(sizes_tokens)) - 1)]
        if sizes_tokens else 0
    )
    std_chars = statistics.pstdev(sizes_chars) if len(sizes_chars) > 1 else 0
    cv = (std_chars / mean_chars) if mean_chars else 0

    # Overlap can be calculated exactly for our character-based chunkers.
    span_data = _merge_spans(chunks, len(source_text))
    if span_data:
        spans, merged = span_data
        covered_chars = sum(end - start for start, end in merged)
        duplicated_chars = max(0, sum(end - start for start, end in spans) - covered_chars)
        overlap_ratio = duplicated_chars / total_chunk_chars if total_chunk_chars else 0
        source_coverage = covered_chars / len(source_text) if source_text else 0
        overlap_method = "Exact from source offsets"
    else:
        overlap_ratio = _adjacent_text_overlap(chunks)
        source_coverage = None
        overlap_method = "Estimated from adjacent chunk text"

    # Boundary integrity: count chunk boundaries that do not split a word.
    boundary_checks = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        start, end = meta.get("start"), meta.get("end")
        if isinstance(start, int) and isinstance(end, int):
            if 0 < start < len(source_text):
                boundary_checks.append(
                    not (source_text[start - 1].isalnum() and source_text[start].isalnum())
                )
            if 0 < end < len(source_text):
                boundary_checks.append(
                    not (source_text[end - 1].isalnum() and source_text[end].isalnum())
                )
    boundary_integrity = (
        sum(boundary_checks) / len(boundary_checks) if boundary_checks else None
    )

    # Heading preservation: markdown headings and common legal section headings.
    heading_pattern = re.compile(
        r"(?m)^(?:#{1,6}\s+.+|(?:SECTION|Section|CHAPTER|Chapter|ARTICLE|Article)\s+[\w.-]+.*)$"
    )
    headings = [m.group(0).strip() for m in heading_pattern.finditer(source_text)]
    if headings:
        combined_chunk_text = "\n".join(texts)
        retained_headings = sum(
            1 for heading in headings
            if heading in combined_chunk_text
            or any(
                heading in str(c.get("metadata", {}).get("headings", ""))
                for c in chunks
            )
        )
        structure_preservation = retained_headings / len(headings)
    else:
        structure_preservation = None

    # Ground-truth table evaluation: original Markdown tables are authoritative.
    table_metrics = evaluate_tables_against_ground_truth(source_text, chunks, target_chunk_size)
    table_integrity = table_metrics["row_integrity"]

    # Metadata completeness: percentage of chunks carrying at least one useful
    # source-location/structure field, excluding the synthetic chunk index.
    useful_meta_keys = {
        "start", "end", "page_numbers", "headings", "source_name",
        "source_uri", "page_number", "section", "section_title",
    }
    metadata_flags = [
        any(_nonempty(c.get("metadata", {}).get(key)) for key in useful_meta_keys)
        for c in chunks
    ]
    metadata_completeness = (
        sum(metadata_flags) / len(metadata_flags) if metadata_flags else 0
    )

    empty_rate = (count - len(nonempty_texts)) / count if count else 0
    redundancy_ratio = (
        max(0, total_chunk_chars - len(source_text)) / total_chunk_chars
        if total_chunk_chars else 0
    )

    return {
        "chunk_count": count,
        "average_chars": mean_chars,
        "median_tokens": median_tokens,
        "p95_tokens": p95_tokens,
        "size_std_chars": std_chars,
        "size_cv": cv,
        "overlap_ratio": overlap_ratio,
        "overlap_method": overlap_method,
        "source_coverage": source_coverage,
        "empty_rate": empty_rate,
        "boundary_integrity": boundary_integrity,
        "structure_preservation": structure_preservation,
        "metadata_completeness": metadata_completeness,
        "table_integrity": table_integrity,
        "table_metrics": table_metrics,
        "redundancy_ratio": redundancy_ratio,
        "sizes_tokens": sizes_tokens,
    }


def _metric_label(value, suffix="%"):
    return "N/A" if value is None else f"{value * 100:.1f}{suffix}"


def render_quality_dashboard(results, source_text):
    """Render cross-experiment metric comparison at the top of Experiment Results."""
    usable = [
        (key, result)
        for key, result in results.items()
        if result.get("chunks") and not result.get("error")
    ]
    if not usable:
        st.info("Run at least one successful experiment to calculate chunk-quality metrics.")
        return

    metric_rows = []
    metric_map = {}
    for result_id, result in usable:
        metrics = calculate_chunk_metrics(
            source_text, result["chunks"],
            target_chunk_size=result.get("config", {}).get("chunk_size")
        )
        metric_map[result_id] = metrics
        metric_rows.append({
            "Experiment": result["name"],
            "Strategy": result["strategy"],
            "Chunks": metrics["chunk_count"],
            "Avg chars": round(metrics["average_chars"]),
            "Median tokens": round(metrics["median_tokens"]),
            "P95 tokens": round(metrics["p95_tokens"]),
            "Overlap": _metric_label(metrics["overlap_ratio"]),
            "Boundary": _metric_label(metrics["boundary_integrity"]),
            "Structure": _metric_label(metrics["structure_preservation"]),
            "Metadata": _metric_label(metrics["metadata_completeness"]),
            "Table detection": _metric_label(metrics["table_metrics"]["table_detection_recall"]),
            "Table headers": _metric_label(metrics["table_metrics"]["header_preservation"]),
            "Table rows": _metric_label(metrics["table_metrics"]["row_preservation"]),
            "Table cells": _metric_label(metrics["table_metrics"]["cell_preservation"]),
            "Rows intact": _metric_label(metrics["table_metrics"]["row_integrity"]),
            "Complete tables": _metric_label(metrics["table_metrics"]["complete_table_integrity"]),
            "Header association": _metric_label(metrics["table_metrics"]["header_association"]),
            "Empty": _metric_label(metrics["empty_rate"]),
            "Redundancy": _metric_label(metrics["redundancy_ratio"]),
        })

    st.subheader("Chunk quality overview")
    st.caption(
        "Automatically calculated from the source Markdown and generated chunks. "
        "No LLM calls or retrieval evaluation are used."
    )

    # Compact high-level indicators across all successful experiments.
    structure_candidates = [
        (m["structure_preservation"], name)
        for name, m in [
            (result["name"], metric_map[key]) for key, result in usable
        ]
        if m["structure_preservation"] is not None
    ]
    best_structure = max(structure_candidates, default=(None, None))
    metadata_candidates = [
        (m["metadata_completeness"], name)
        for name, m in [
            (result["name"], metric_map[key]) for key, result in usable
        ]
    ]
    best_metadata = max(metadata_candidates, default=(None, None))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Experiments evaluated", len(usable))
    k2.metric(
        "Best structure preservation",
        _metric_label(best_structure[0]),
        best_structure[1] or "Not measurable",
    )
    k3.metric(
        "Best metadata completeness",
        _metric_label(best_metadata[0]),
        best_metadata[1] or "—",
    )
    avg_empty = statistics.mean(m["empty_rate"] for m in metric_map.values())
    k4.metric("Average empty chunk rate", _metric_label(avg_empty))

    st.markdown("#### Experiment comparison")
    st.dataframe(metric_rows, use_container_width=True, hide_index=True)

    with st.expander("Ground-truth table preservation analysis", expanded=True):
        gt_tables = extract_ground_truth_tables(source_text)
        st.caption(
            f"Ground truth: {len(gt_tables)} Markdown tables extracted from the original uploaded document. "
            "Each experiment is compared against their headers, rows and cells."
        )
        if not gt_tables:
            st.warning("No Markdown pipe tables were detected in the original document.")
        else:
            table_summary = []
            for result_id, result in usable:
                tm = metric_map[result_id]["table_metrics"]
                table_summary.append({
                    "Experiment": result["name"],
                    "Tables detected": f'{tm["table_detection_recall"] * 100:.1f}%' if tm["table_detection_recall"] is not None else "N/A",
                    "Headers preserved": _metric_label(tm["header_preservation"]),
                    "Rows preserved": _metric_label(tm["row_preservation"]),
                    "Cells preserved": _metric_label(tm["cell_preservation"]),
                    "Rows intact in one chunk": _metric_label(tm["row_integrity"]),
                    "Complete tables in one chunk": _metric_label(tm["complete_table_integrity"]),
                    "Header association": _metric_label(tm["header_association"]),
                    "Average chunks touched/table": round(tm["fragmentation"], 2) if tm["fragmentation"] is not None else "N/A",
                    "Tables larger than target": tm["oversized_table_count"] if tm["oversized_table_count"] is not None else "N/A",
                })
            st.dataframe(table_summary, use_container_width=True, hide_index=True)

            selected_exp = st.selectbox(
                "Inspect table-level results",
                options=[(key, result["name"]) for key, result in usable],
                format_func=lambda item: item[1],
                key="table_gt_inspect_exp",
            )
            selected_metrics = metric_map[selected_exp[0]]["table_metrics"]
            st.dataframe(selected_metrics["details"], use_container_width=True, hide_index=True)
            st.caption(
                "A row counts as preserved when all its normalized cell values remain in the chunk output. "
                "A row counts as intact only when all its cells occur together in at least one chunk. "
                "Repeated/common cell values can cause false matches; inspect low-confidence or unusual cases manually."
            )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Chunk size distribution")
        size_chart = {
            result["name"]: [
                round(metric_map[key]["median_tokens"]),
                round(metric_map[key]["p95_tokens"]),
            ]
            for key, result in usable
        }
        st.bar_chart(
            {
                "Median tokens": {
                    name: values[0] for name, values in size_chart.items()
                },
                "P95 tokens": {
                    name: values[1] for name, values in size_chart.items()
                },
            }
        )
    with chart_right:
        st.markdown("#### Structural quality indicators")
        quality_chart = {
            result["name"]: {
                "Boundary integrity": (
                    round(metric_map[key]["boundary_integrity"] * 100, 1)
                    if metric_map[key]["boundary_integrity"] is not None else 0
                ),
                "Structure preservation": (
                    round(metric_map[key]["structure_preservation"] * 100, 1)
                    if metric_map[key]["structure_preservation"] is not None else 0
                ),
                "Metadata completeness": round(
                    metric_map[key]["metadata_completeness"] * 100, 1
                ),
            }
            for key, result in usable
        }
        st.bar_chart(quality_chart)

    with st.expander("Metric definitions and limitations"):
        st.markdown(
            """
            - **Average chars / Median tokens / P95 tokens:** chunk-size distribution.
            - **Overlap:** exact duplicate character coverage when source offsets exist; otherwise estimated from matching adjacent chunk text.
            - **Boundary integrity:** percentage of measurable chunk boundaries that do not split an alphanumeric word. Only available when source offsets are supplied.
            - **Structure preservation:** percentage of detected Markdown/legal headings found in generated chunks or heading metadata.
            - **Metadata completeness:** percentage of chunks containing at least one useful source-location or structure field.
            - **Table metrics:** original Markdown pipe tables are ground truth. Row preservation checks whether cell values survive anywhere; row integrity checks whether each row stays together; complete table integrity checks whether the header and every row stay together in one chunk; header association checks whether rows remain with their header.
            - **Empty rate:** percentage of chunks containing no non-whitespace text.
            - **Redundancy:** estimated extra chunk characters beyond source length. This is a coarse duplication indicator, not a semantic redundancy score.
            - **Source coverage:** percentage of source characters covered by chunks; available when source offsets exist.
            - **Size CV:** standard deviation divided by mean chunk size; lower means more uniform sizes, not necessarily better chunks.

            N/A means the source representation or chunk metadata does not support a reliable calculation. These are structural diagnostics, not measures of semantic correctness or retrieval quality.
            """
        )

    return metric_map

def show_experiment_results(text):
    results = st.session_state.get("experiment_results", {})
    if not results:
        st.info("Run your configured experiments to see their results here.")
        return

    render_quality_dashboard(results, text)

    ordered_ids = [
        item["id"] for item in st.session_state.experiment_configs
        if item["id"] in results
    ]
    # Keep completed results visible even if a config was subsequently removed.
    ordered_ids += [key for key in results if key not in ordered_ids]
    result_tabs = st.tabs(["Original document"] + [results[key]["name"] for key in ordered_ids])

    with result_tabs[0]:
        st.subheader(st.session_state.get("filename", "Original parsed Markdown"))
        st.markdown(text)

    for tab, result_id in zip(result_tabs[1:], ordered_ids):
        result = results[result_id]
        with tab:
            st.title(result["name"])
            st.caption(
                f"Strategy: {result['strategy']} · "
                f"Completed chunks: {len(result['chunks'])}"
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Chunks", len(result["chunks"]))
            m2.metric(
                "Average chunk size",
                f"{int(sum(len(c['text']) for c in result['chunks']) / len(result['chunks'])) if result['chunks'] else 0:,} chars",
            )
            m3.metric(
                "Total chunk characters",
                f"{sum(len(c['text']) for c in result['chunks']):,}",
            )
            st.json(result["config"])
            if result.get("serialized_document") is not None:
                ext = "html" if result["config"].get("serialization_format") == "html" else "md"
                st.download_button(
                    "Download serialized document",
                    result["serialized_document"],
                    file_name=f"{result['name']}_serialized.{ext}",
                    key=f"serialized_{result_id}",
                )
            st.download_button(
                "Download chunks JSON",
                json.dumps(result["chunks"], ensure_ascii=False, indent=2, default=str),
                file_name=f"{result['name']}_chunks.json",
                mime="application/json",
                key=f"download_{result_id}",
            )
            if result.get("error"):
                st.error(f"Experiment failed: {result['error']}")
            elif not result["chunks"]:
                st.warning("This strategy returned no chunks.")
            else:
                # Full-width result view: one selected experiment fills the workspace.
                for index, chunk in enumerate(result["chunks"]):
                    st.markdown(
                        f"### Chunk {index + 1} · {len(chunk['text']):,} characters "
                        f"· ~{approx_tokens(chunk['text']):,} tokens"
                    )
                    st.markdown(chunk["text"])
                    with st.expander(f"Metadata — Chunk {index + 1}"):
                        st.json(chunk["metadata"])
                    if index < len(result["chunks"]) - 1:
                        st.divider()


ensure_experiment_state()

st.title("Legal RAG — Chunking Playground")
st.caption(
    "Upload your already-parsed Markdown (.md) document. "
    "Docling strategies convert the Markdown to a document before chunking."
)

# The original single-run playground remains as its own tab.
playground_tab, experiments_tab = st.tabs(["Chunking Playground", "Experiments"])

with st.sidebar:
    uploaded = st.file_uploader("Upload parsed Markdown document", type=["md"], key="source_upload")

if uploaded is not None:
    try:
        text = extract_text(uploaded)
        if st.session_state.get("active_upload") != uploaded.name:
            st.session_state.pop("chunks", None)
            st.session_state.pop("serialized_document", None)
            st.session_state["experiment_results"] = {}
            st.session_state["active_upload"] = uploaded.name
        st.session_state["source_text"] = text
        st.session_state["filename"] = uploaded.name
    except Exception as exc:
        st.error(f"Could not read the Markdown file: {exc}")
        st.stop()
else:
    text = st.session_state.get("source_text", "")

with st.sidebar:
    st.caption("The uploaded Markdown is shared by both tabs.")

with playground_tab:
    st.subheader("Single-run chunking")
    strategy = st.selectbox(
        "Chunking strategy",
        ALL_STRATEGIES,
        key="single_strategy",
    )
    single_config = {}
    if strategy in {"Fixed-size", "Recursive", "Structure-aware"}:
        size_col, overlap_col = st.columns(2)
        single_config["chunk_size"] = size_col.number_input(
            "Target size (characters)", min_value=100, max_value=20000,
            value=1000, step=100, key="single_size",
        )
        single_config["overlap"] = overlap_col.number_input(
            "Overlap (characters)", min_value=0, max_value=10000,
            value=100, step=10, key="single_overlap",
        )
    elif strategy in {"Hybrid", "Advanced"}:
        single_config["tokenizer_model"] = st.text_input(
            "Hugging Face tokenizer model (optional)", key="single_tokenizer"
        )
        single_config["max_tokens"] = st.number_input(
            "Maximum tokens", min_value=64, max_value=8192,
            value=512, step=64, key="single_max_tokens",
        )
        if strategy == "Advanced":
            single_config["serialization_format"] = st.selectbox(
                "Serialized document format", ["markdown", "html"], key="single_format"
            )
            single_config["triplet_tables"] = st.checkbox(
                "Serialize tables as triplets", key="single_triplet",
                disabled=single_config["serialization_format"] != "markdown",
            )
            single_config["image_placeholder"] = st.text_input(
                "Image placeholder", value="<!-- image -->", key="single_image_placeholder"
            )
    elif strategy == "Line-based token":
        single_config["prefix"] = st.text_input("Chunk prefix (optional)", key="single_prefix")

    run_single = st.button("Generate chunks", type="primary", key="run_single")
    if run_single:
        if not text.strip():
            st.warning("Upload a parsed Markdown (.md) document first.")
        elif (
            strategy in {"Fixed-size", "Recursive", "Structure-aware"}
            and single_config["overlap"] >= single_config["chunk_size"]
        ):
            st.error("Overlap must be smaller than chunk size.")
        else:
            try:
                chunks, serialized = execute_strategy(
                    strategy, text, st.session_state.get("filename", "document.md"), single_config
                )
                st.session_state["chunks"] = chunks
                st.session_state["strategy"] = strategy
                st.session_state["serialized_document"] = serialized
                st.session_state["single_serialization_format"] = single_config.get("serialization_format")
            except Exception as exc:
                st.error(f"Chunking failed: {exc}")

    chunks = st.session_state.get("chunks", [])
    if not text:
        st.info("Upload your parsed Markdown (.md) document using the sidebar.")
    else:
        st.subheader(st.session_state.get("filename", "Parsed Markdown document"))
        a, b, c, d = st.columns(4)
        a.metric("Characters", f"{len(text):,}")
        b.metric("Approx. tokens", f"{approx_tokens(text):,}")
        c.metric("Chunks", len(chunks))
        avg_size = int(sum(len(x["text"]) for x in chunks) / len(chunks)) if chunks else 0
        d.metric("Average chunk size", f"{avg_size:,} chars")
        if chunks:
            st.caption("Synchronized scrolling is ON — scroll either panel and the other follows.")
            render_comparison(text, chunks, st.session_state.get("strategy", "Chunking result"))
            st.download_button(
                "Download all chunks as JSON",
                json.dumps(chunks, ensure_ascii=False, indent=2, default=str),
                file_name="chunks.json",
                mime="application/json",
                key="single_download",
            )
            if st.session_state.get("serialized_document") is not None:
                ext = "html" if st.session_state.get("single_serialization_format") == "html" else "md"
                st.download_button(
                    "Download serialized document",
                    st.session_state["serialized_document"],
                    file_name=f"serialized_document.{ext}",
                    key="single_serialized_download",
                )

with experiments_tab:
    st.header("Experiment configuration")
    st.caption(
        "Configure several chunking runs, execute them together, then switch between "
        "the original document and each experiment's full-width results."
    )

    top_a, top_b = st.columns([1, 1])
    with top_a:
        if st.button("＋ Add experiment", key="add_experiment"):
            st.session_state.experiment_configs.append(
                new_experiment(len(st.session_state.experiment_configs) + 1)
            )
            st.rerun()
    with top_b:
        if st.button("Reset examples", key="reset_experiments"):
            st.session_state.experiment_configs = [
                {"id": "exp001", "name": "EXP-01", "strategy": "Fixed-size"},
                {"id": "exp002", "name": "EXP-02", "strategy": "Fixed-size"},
                {"id": "exp003", "name": "EXP-03", "strategy": "Recursive"},
            ]
            st.session_state.experiment_results = {}
            st.rerun()

    configs_to_run = []
    for index, config in enumerate(st.session_state.experiment_configs):
        st.subheader(f"Experiment {index + 1}")
        values = render_experiment_config(config, index)
        values["id"] = config["id"]
        configs_to_run.append(values)

    run_all = st.button("▶ Run all experiments", type="primary", key="run_all_experiments")
    if run_all:
        if not text.strip():
            st.error("Upload a parsed Markdown (.md) document before running experiments.")
        else:
            invalid = [
                item["name"] for item in configs_to_run
                if item.get("strategy") in {"Fixed-size", "Recursive", "Structure-aware"}
                and item.get("overlap", 0) >= item.get("chunk_size", 1)
            ]
            if invalid:
                st.error(
                    "Overlap must be smaller than chunk size for: " + ", ".join(invalid)
                )
            else:
                progress = st.progress(0, text="Preparing experiments...")
                status = st.empty()
                results = dict(st.session_state.experiment_results)
                total = len(configs_to_run)
                for index, config in enumerate(configs_to_run):
                    name = config["name"].strip() or f"EXP-{index + 1:02d}"
                    status.write(f"Running {name}: {config['strategy']}...")
                    try:
                        chunks, serialized = execute_strategy(
                            config["strategy"],
                            text,
                            st.session_state.get("filename", "document.md"),
                            config,
                        )
                        results[config["id"]] = {
                            "name": name,
                            "strategy": config["strategy"],
                            "config": {
                                key: value for key, value in config.items()
                                if key not in {"id", "name", "strategy"}
                            },
                            "chunks": chunks,
                            "serialized_document": serialized,
                        }
                    except Exception as exc:
                        results[config["id"]] = {
                            "name": name,
                            "strategy": config["strategy"],
                            "config": {
                                key: value for key, value in config.items()
                                if key not in {"id", "name", "strategy"}
                            },
                            "chunks": [],
                            "serialized_document": None,
                            "error": str(exc),
                        }
                    progress.progress(
                        (index + 1) / total,
                        text=f"Completed {index + 1} of {total} experiments",
                    )
                st.session_state.experiment_results = results
                status.success("Experiment run finished. Open the results below.")

    # Persist edits to the configuration model after widget values have been read.
    st.session_state.experiment_configs = [
        {
            "id": values["id"],
            "name": values["name"],
            "strategy": values["strategy"],
        }
        for values in configs_to_run
    ]

    st.divider()
    st.header("Experiment results")
    if not text:
        st.info("Upload a Markdown document to configure and run experiments.")
    else:
        show_experiment_results(text)
