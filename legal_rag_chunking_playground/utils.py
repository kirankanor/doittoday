import html


def extract_text(file):
    """Read an already-parsed Markdown file without running a parser."""
    if not getattr(file, "name", "").lower().endswith(".md"):
        raise ValueError("Please upload a Markdown (.md) file.")
    return file.getvalue().decode("utf-8-sig")


def approx_tokens(text):
    return round(len(text.split()) * 1.3) if text.strip() else 0


def render_highlight(chunks, selected):
    blocks = []
    for i, item in enumerate(chunks):
        safe = html.escape(item["text"]).replace("\\n", "<br>")
        bg = "#fff0b3" if i == selected else "#f1f3f5"
        border = "#d18b00" if i == selected else "#aab2bd"
        blocks.append(
            f'<div style="background:{bg};border-left:5px solid {border};'
            f'padding:10px;margin:6px 0;border-radius:4px">{safe}</div>'
        )
    return "".join(blocks)
