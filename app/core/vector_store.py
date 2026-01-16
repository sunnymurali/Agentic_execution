"""
Vector store client initialization
"""

from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.core.config import azure_config, qdrant_config


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Get Azure OpenAI embeddings instance"""
    return AzureOpenAIEmbeddings(
        api_key=azure_config.api_key,
        azure_endpoint=azure_config.endpoint,
        model=azure_config.embedding_model
    )


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance"""
    return QdrantClient(host=qdrant_config.host, port=qdrant_config.port)


def get_vector_store(client: QdrantClient, embeddings: AzureOpenAIEmbeddings) -> QdrantVectorStore:
    """Get Qdrant vector store instance"""
    return QdrantVectorStore(
        client=client,
        collection_name=qdrant_config.collection_name,
        embedding=embeddings
    )


def ensure_collection_exists(client: QdrantClient):
    """Create collection if it doesn't exist"""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if qdrant_config.collection_name not in collection_names:
        client.create_collection(
            collection_name=qdrant_config.collection_name,
            vectors_config=VectorParams(
                size=qdrant_config.vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {qdrant_config.collection_name}")
    else:
        print(f"Collection exists: {qdrant_config.collection_name}")
