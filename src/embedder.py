import os
import sys
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="rag-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
)


def generate_embedding(text: str) -> list:
    response = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text
    )
    return response.data[0].embedding


def embed_and_upload(chunks: list) -> None:
    total = len(chunks)
    print(f"Embedding and uploading {total} chunks...")

    batch = []
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = generate_embedding(chunk["content"])
        batch.append(chunk)

        # upload in batches of 10
        if len(batch) == 10:
            search_client.upload_documents(documents=batch)
            print(f"Uploaded {i + 1}/{total} chunks")
            batch = []
            time.sleep(0.5)  # avoid rate limiting

    # upload remaining
    if batch:
        search_client.upload_documents(documents=batch)
        print(f"Uploaded {total}/{total} chunks")

    print(f"\nDone. All {total} chunks embedded and indexed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/embedder.py <path_to_file>")
        sys.exit(1)

    sys.path.append(os.path.dirname(__file__))
    from chunker import chunk_document

    file_path = sys.argv[1]
    print(f"Processing: {file_path}")

    chunks = chunk_document(file_path)
    print(f"Total chunks to embed: {len(chunks)}")

    embed_and_upload(chunks)