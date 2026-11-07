from docling.document_converter import DocumentConverter


def load_document(source: str):
    """
    Parse a document using Docling. Accepts a local file path or URL.

    Returns:
        Docling document object with structure metadata
    """
    print(f"  Parsing: {source}")
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document
