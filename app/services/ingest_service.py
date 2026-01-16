"""
Document ingestion service implementation
"""

from typing import Dict, Any
from datetime import datetime, timezone

from app.services.interfaces import IIngestService, IVectorStore
from app.helpers.document_helper import (
    generate_document_id,
    get_document_name,
    build_document_path,
    read_pdf_text
)
from app.helpers.chunking_helper import chunk_document
from app.helpers.validation_helper import validate_file_exists


class IngestService(IIngestService):
    """Concrete implementation of IIngestService"""

    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store

    def ingest(self, document_id: str, folder_path: str, document_type: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Ingest a document into vector store

        Args:
            document_id: Document identifier (filename without extension)
            folder_path: Folder containing the document
            document_type: Type of document (earnings, prospectus, 10k, 10q, etc.)
            metadata: Optional additional metadata

        Returns:
            Dict with ingestion results
        """
        print(f"=== INGEST SERVICE ===")
        print(f"Document ID: {document_id}")
        print(f"Document Type: {document_type}")
        print(f"Folder Path: {folder_path}")

        # Build full document path
        document_path = build_document_path(folder_path, document_id, ".pdf")
        print(f"Full Path: {document_path}")

        # Validate file exists
        if not validate_file_exists(document_path):
            return {
                "document_id": document_id,
                "status": "error",
                "message": f"File not found: {document_path}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Check if already indexed
        if self.check_exists(document_id):
            print(f"Document already indexed, skipping...")
            return {
                "document_id": document_id,
                "status": "skipped",
                "message": "Document already indexed",
                "chunk_count": 0,
                "already_indexed": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Read PDF text
        print("Loading document...")
        text = read_pdf_text(document_path)
        print(f"Extracted {len(text):,} characters")

        # Chunk document
        print("Chunking document...")
        document_name = get_document_name(document_path)
        chunks = chunk_document(text, document_id, document_name, document_path, document_type)
        print(f"Created {len(chunks)} chunks")

        # Store in vector database
        print("Generating embeddings and storing...")
        count = self.vector_store.add_documents(chunks, document_id)
        print(f"Stored {count} chunks in vector database")

        return {
            "document_id": document_id,
            "document_name": document_name,
            "status": "success",
            "message": "Document ingested successfully",
            "chunk_count": count,
            "character_count": len(text),
            "already_indexed": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def check_exists(self, document_id: str) -> bool:
        """Check if document already exists in vector store"""
        return self.vector_store.exists(document_id)
