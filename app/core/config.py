"""
Configuration settings for the application
"""

import os
from dataclasses import dataclass


@dataclass
class AzureConfig:
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/")
    api_version: str = "2025-01-01-preview"
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4.1"


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "document_store"
    vector_size: int = 3072  # text-embedding-3-large dimension


@dataclass
class ChunkingConfig:
    chunk_size: int = 2500
    chunk_overlap: int = 200
    separators: list = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = [
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]


# Global config instances
azure_config = AzureConfig()
qdrant_config = QdrantConfig()
chunking_config = ChunkingConfig()
