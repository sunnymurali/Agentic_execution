"""
Validation helper functions
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


def check_document_exists(client: QdrantClient, collection_name: str, document_id: str) -> bool:
    """Check if document already exists in Qdrant"""
    try:
        result = client.scroll(
            collection_name=collection_name,
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


def validate_file_exists(file_path: str) -> bool:
    """Check if file exists at path"""
    return os.path.exists(file_path)


def validate_folder_exists(folder_path: str) -> bool:
    """Check if folder exists"""
    return os.path.isdir(folder_path)
