"""
gRPC Retrieval Server - Provides vector search capabilities over gRPC
"""

import grpc
from concurrent import futures
from typing import Optional

from app.grpc import retrieval_pb2
from app.grpc import retrieval_pb2_grpc
from app.services.vector_store_service import QdrantVectorStoreService
from app.core.vector_store import (
    get_qdrant_client,
    get_embeddings,
    get_vector_store as get_langchain_vector_store,
    ensure_collection_exists
)


class RetrievalServicer(retrieval_pb2_grpc.RetrievalServiceServicer):
    """gRPC service implementation for document retrieval"""

    def __init__(self, vector_store: QdrantVectorStoreService):
        self.vector_store = vector_store
        print("[gRPC Server] RetrievalServicer initialized")

    def Search(self, request, context):
        """Single query search"""
        try:
            print(f"[gRPC] Search request - query: '{request.query[:50]}...', doc_id: {request.document_id}")

            results = self.vector_store.search(
                query=request.query,
                document_id=request.document_id,
                document_type=request.document_type,
                k=request.k or 10
            )

            chunks = []
            for doc, score in results:
                chunk = retrieval_pb2.Chunk(
                    content=doc.page_content,
                    score=score,
                    metadata={k: str(v) for k, v in doc.metadata.items()},
                    query=request.query
                )
                chunks.append(chunk)

            print(f"[gRPC] Search returned {len(chunks)} chunks")

            return retrieval_pb2.SearchResponse(
                chunks=chunks,
                total_chunks=len(chunks),
                status="success",
                message=""
            )

        except Exception as e:
            print(f"[gRPC] Search error: {e}")
            return retrieval_pb2.SearchResponse(
                chunks=[],
                total_chunks=0,
                status="error",
                message=str(e)
            )

    def MultiSearch(self, request, context):
        """Multi-query search with optional deduplication"""
        try:
            print(f"[gRPC] MultiSearch request - {len(request.queries)} queries, doc_id: {request.document_id}")

            all_chunks = []
            seen_content = set()

            for query in request.queries:
                results = self.vector_store.search(
                    query=query,
                    document_id=request.document_id,
                    document_type=request.document_type,
                    k=request.k_per_query or 10
                )

                for doc, score in results:
                    content = doc.page_content

                    # Deduplicate if requested
                    if request.deduplicate:
                        content_hash = hash(content[:200])
                        if content_hash in seen_content:
                            continue
                        seen_content.add(content_hash)

                    chunk = retrieval_pb2.Chunk(
                        content=content,
                        score=score,
                        metadata={k: str(v) for k, v in doc.metadata.items()},
                        query=query
                    )
                    all_chunks.append(chunk)

            print(f"[gRPC] MultiSearch returned {len(all_chunks)} chunks")

            return retrieval_pb2.SearchResponse(
                chunks=all_chunks,
                total_chunks=len(all_chunks),
                status="success",
                message=""
            )

        except Exception as e:
            print(f"[gRPC] MultiSearch error: {e}")
            return retrieval_pb2.SearchResponse(
                chunks=[],
                total_chunks=0,
                status="error",
                message=str(e)
            )

    def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            # Try a simple operation to verify vector store is working
            return retrieval_pb2.HealthCheckResponse(
                status="healthy",
                vector_store="connected"
            )
        except Exception as e:
            return retrieval_pb2.HealthCheckResponse(
                status="unhealthy",
                vector_store=f"error: {e}"
            )


def create_server(
    port: int = 50051,
    max_workers: int = 10,
    vector_store: Optional[QdrantVectorStoreService] = None
) -> grpc.Server:
    """Create and configure the gRPC server"""

    # Initialize vector store if not provided
    if vector_store is None:
        print("[gRPC Server] Initializing vector store...")
        client = get_qdrant_client()
        ensure_collection_exists(client)
        embeddings = get_embeddings()
        langchain_store = get_langchain_vector_store(client, embeddings)
        vector_store = QdrantVectorStoreService(client, langchain_store)

    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # Add servicer
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
        RetrievalServicer(vector_store),
        server
    )

    # Bind to port
    server.add_insecure_port(f'[::]:{port}')

    return server


def serve(port: int = 50051):
    """Start the gRPC server"""
    server = create_server(port=port)
    server.start()

    print(f"\n{'='*50}")
    print(f"gRPC Retrieval Server started on port {port}")
    print(f"{'='*50}\n")

    server.wait_for_termination()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    serve(port=50051)
