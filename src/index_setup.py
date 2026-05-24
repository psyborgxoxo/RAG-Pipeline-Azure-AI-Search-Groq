import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = "rag-index"

def create_index():
    client = SearchIndexClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY)
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="default-profile"
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SimpleField(
            name="doc_type",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SimpleField(
            name="date",
            type=SearchFieldDataType.String,
            filterable=True
        ),
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="default-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")
        ]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    result = client.create_or_update_index(index)
    print(f"Index '{result.name}' created successfully.")

if __name__ == "__main__":
    create_index()