import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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


def embed_query(text: str) -> list:
    response = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text
    )
    return response.data[0].embedding


def hybrid_search(query: str, top_k: int = 5) -> list:
    query_vector = embed_query(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content", "source", "doc_type"],
        top=top_k
    )

    chunks = []
    for result in results:
        chunks.append({
            "id":       result["id"],
            "content":  result["content"],
            "source":   result["source"],
            "doc_type": result["doc_type"],
            "score":    result["@search.score"]
        })

    return chunks


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is complexity in closed systems?"

    print(f"Query: {query}")
    print("-" * 60)

    results = hybrid_search(query)

    for i, chunk in enumerate(results):
        print(f"\nResult {i+1} | Score: {chunk['score']:.4f} | Source: {chunk['source']}")
        print(chunk["content"][:300])
        print("-" * 60)