import os
import sys
from dotenv import load_dotenv
from groq import Groq
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

load_dotenv()

# Clients
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="rag-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
)

embedding_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

chat_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions 
based strictly on the provided context. 
Always base your answer only on the context below.
If the answer is not in the context, say 'I could not find this in the provided documents.'
Never make up information."""


def embed_query(text: str) -> list:
    response = embedding_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text
    )
    return response.data[0].embedding


def retrieve_chunks(query: str, top_k: int = 5) -> list:
    vector = embed_query(query)
    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        select=["id", "content", "source"],
        top=top_k
    )
    return list(results)


def generate_answer(query: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )

    response = chat_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message.content


def ask(query: str) -> dict:
    chunks   = retrieve_chunks(query)
    answer   = generate_answer(query, chunks)
    sources  = list(set(c["source"] for c in chunks))

    return {
        "question": query,
        "answer":   answer,
        "sources":  sources
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What is the coffee automaton?"

    print(f"Question: {query}")
    print("-" * 60)

    result = ask(query)

    print(f"Answer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")