"""
Chunking helper functions
"""

from datetime import datetime, timezone
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import chunking_config


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Get configured text splitter"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
        separators=chunking_config.separators,
        length_function=len
    )


def chunk_text(text: str) -> List[str]:
    """Chunk text into smaller pieces"""
    splitter = get_text_splitter()
    return splitter.split_text(text)


def chunk_document(
    text: str,
    document_id: str,
    document_name: str,
    document_path: str,
    document_type: str
) -> List[Document]:
    """Chunk text and create Document objects with metadata"""
    splitter = get_text_splitter()

    # Create a document for splitting
    doc = Document(page_content=text, metadata={"source": document_path})
    chunks_docs = splitter.split_documents([doc])

    # Add metadata to each chunk
    for i, chunk_doc in enumerate(chunks_docs):
        chunk_doc.metadata.update({
            "chunk_id": i,
            "document_id": document_id,
            "document_name": document_name,
            "document_path": document_path,
            "document_type": document_type,
            "total_chunks": len(chunks_docs),
            "indexed_at": datetime.now(timezone.utc).isoformat()
        })

    return chunks_docs
