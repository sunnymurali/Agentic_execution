"""
gRPC Retrieval Client - Client for calling the retrieval service
"""

import grpc
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.grpc import retrieval_pb2
from app.grpc import retrieval_pb2_grpc


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk"""
    content: str
    score: float
    metadata: Dict[str, str]
    query: str


class RetrievalClient:
    """gRPC client for the retrieval service"""

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None

    def connect(self):
        """Establish connection to the gRPC server"""
        if self.channel is None:
            self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
            self.stub = retrieval_pb2_grpc.RetrievalServiceStub(self.channel)
            print(f"[gRPC Client] Connected to {self.host}:{self.port}")

    def close(self):
        """Close the connection"""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def search(
        self,
        query: str,
        document_id: str,
        document_type: str,
        k: int = 10
    ) -> List[RetrievedChunk]:
        """
        Single query search.

        Args:
            query: Search query
            document_id: Document to search in
            document_type: Type of document
            k: Number of results

        Returns:
            List of RetrievedChunk objects
        """
        if self.stub is None:
            self.connect()

        request = retrieval_pb2.SearchRequest(
            query=query,
            document_id=document_id,
            document_type=document_type,
            k=k
        )

        response = self.stub.Search(request)

        if response.status == "error":
            raise Exception(f"Retrieval error: {response.message}")

        return [
            RetrievedChunk(
                content=chunk.content,
                score=chunk.score,
                metadata=dict(chunk.metadata),
                query=chunk.query
            )
            for chunk in response.chunks
        ]

    def multi_search(
        self,
        queries: List[str],
        document_id: str,
        document_type: str,
        k_per_query: int = 10,
        deduplicate: bool = True
    ) -> List[RetrievedChunk]:
        """
        Multi-query search with optional deduplication.

        Args:
            queries: List of search queries
            document_id: Document to search in
            document_type: Type of document
            k_per_query: Results per query
            deduplicate: Remove duplicate chunks

        Returns:
            List of RetrievedChunk objects
        """
        if self.stub is None:
            self.connect()

        request = retrieval_pb2.MultiSearchRequest(
            queries=queries,
            document_id=document_id,
            document_type=document_type,
            k_per_query=k_per_query,
            deduplicate=deduplicate
        )

        response = self.stub.MultiSearch(request)

        if response.status == "error":
            raise Exception(f"Retrieval error: {response.message}")

        return [
            RetrievedChunk(
                content=chunk.content,
                score=chunk.score,
                metadata=dict(chunk.metadata),
                query=chunk.query
            )
            for chunk in response.chunks
        ]

    def health_check(self) -> Dict[str, str]:
        """Check if the retrieval service is healthy"""
        if self.stub is None:
            self.connect()

        request = retrieval_pb2.HealthCheckRequest()
        response = self.stub.HealthCheck(request)

        return {
            "status": response.status,
            "vector_store": response.vector_store
        }


# Singleton client instance for reuse
_client_instance: Optional[RetrievalClient] = None


def get_retrieval_client(host: str = "localhost", port: int = 50051) -> RetrievalClient:
    """Get or create a retrieval client instance"""
    global _client_instance

    if _client_instance is None:
        _client_instance = RetrievalClient(host=host, port=port)
        _client_instance.connect()

    return _client_instance
