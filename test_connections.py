import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

load_dotenv()

# Test 1 — Azure AI Search
print("Testing Azure AI Search...")
try:
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="rag-index",
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
    )
    results = search_client.search(search_text="test", top=1)
    list(results)
    print("Azure AI Search — CONNECTED")
except Exception as e:
    print(f"Azure AI Search — FAILED: {e}")

# Test 2 — Azure OpenAI Embeddings
print("\nTesting Azure OpenAI Embeddings...")
try:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-01"
    )
    response = client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input="test connection"
    )
    vector = response.data[0].embedding
    print(f"Azure OpenAI Embeddings — CONNECTED")
    print(f"Vector dimensions: {len(vector)}")
except Exception as e:
    print(f"Azure OpenAI Embeddings — FAILED: {e}")

    # Test 3 — Groq API (chat)
print("\nTesting Groq API...")
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        max_tokens=10
    )
    print(f"Groq API — CONNECTED")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Groq API — FAILED: {e}")