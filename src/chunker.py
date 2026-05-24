from langchain_text_splitters import RecursiveCharacterTextSplitter
from etl_parser import extract


def chunk_document(file_path: str, chunk_size: int = 512, overlap: int = 50) -> list:
    doc = extract(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? "]
    )

    texts = splitter.split_text(doc["content"])

    chunks = [
        {
            "id": f"{doc['source'].replace('.', '_').replace(' ', '_')}-chunk{i}",
            "content":  text,
            "source":   doc["source"],
            "doc_type": doc["doc_type"],
            "date":     doc["date"]
        }
        for i, text in enumerate(texts)
    ]

    return chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/chunker.py <path_to_file>")
        sys.exit(1)

    chunks = chunk_document(sys.argv[1])

    print(f"Total chunks : {len(chunks)}")
    print(f"\n--- Chunk 0 ---")
    print(f"ID      : {chunks[0]['id']}")
    print(f"Content : {chunks[0]['content'][:300]}")
    print(f"\n--- Chunk 1 ---")
    print(f"ID      : {chunks[1]['id']}")
    print(f"Content : {chunks[1]['content'][:300]}")