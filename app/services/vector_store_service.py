"""
Vector store service implementation
"""

from typing import List
from uuid import uuid4

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.services.interfaces import IVectorStore
from app.core.config import qdrant_config


class QdrantVectorStoreService(IVectorStore):
    """Qdrant implementation of IVectorStore"""

    def __init__(self, client: QdrantClient, vector_store: QdrantVectorStore):
        self.client = client
        self.vector_store = vector_store
        self.collection_name = qdrant_config.collection_name

    def add_documents(self, documents: List, document_id: str) -> int:
        """Add documents to Qdrant vector store"""
        if not documents:
            return 0

        uuids = [str(uuid4()) for _ in documents]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        return len(documents)

    def search(self, query: str, document_id: str, document_type: str, k: int = 10) -> List:
        """Search for similar documents with document_id and document_type filter"""
        qdrant_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.document_id",
                    match=qdrant_models.MatchValue(value=document_id)
                ),
                qdrant_models.FieldCondition(
                    key="metadata.document_type",
                    match=qdrant_models.MatchValue(value=document_type)
                )
            ]
        )

        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=qdrant_filter
        )

        return results

    def exists(self, document_id: str) -> bool:
        """Check if document exists in Qdrant"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.document_id",
                            match=qdrant_models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1
            )
            return len(result[0]) > 0
        except Exception:
            return False

    def delete(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.document_id",
                                match=qdrant_models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            return True
        except Exception:
            return False
