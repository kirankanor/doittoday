# import sys
# from pathlib import Path

# from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

# from .parsers import load_document

# def parse_and_save() -> None:
#     source = Path(sys.argv[1])
#     document = load_document(str(source))
#     output_dir = Path(__file__).resolve().parents[3] / "results"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     output = output_dir / f"{source.stem}.md"
#     output.write_text(MarkdownDocSerializer(doc=document).serialize().text, encoding="utf-8")
#     print(f"Saved to {output}")


from .chunkers import split_text

# def main():
#     with open("results\docling_benchmark.md" , "r") as f:
#         data = f.read()
#     chunks = split_text(data)

import os
from ..database.vector_store import VectorStore

vs = VectorStore(
    dsn=os.getenv("PG_VECTOR_URL") ,
    embedder="nomic-ai/nomic-embed-text-v1.5" ,
    collection="vector_db" ,
    dimension=768 
)
def main():
    print(vs.list_collections())



if __name__ == "__main__":
    main()
