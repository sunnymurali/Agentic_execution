"""
Service interfaces (Abstract Base Classes)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IVectorStore(ABC):
    """Interface for vector store operations"""

    @abstractmethod
    def add_documents(self, documents: List, document_id: str) -> int:
        """Add documents to vector store, return count added"""
        pass

    @abstractmethod
    def search(self, query: str, document_id: str, document_type: str, k: int) -> List:
        """Search for similar documents filtered by document_id and document_type"""
        pass

    @abstractmethod
    def exists(self, document_id: str) -> bool:
        """Check if document exists in store"""
        pass

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete document from store"""
        pass


class IIngestService(ABC):
    """Interface for document ingestion"""

    @abstractmethod
    def ingest(self, document_id: str, folder_path: str, document_type: str, metadata: Dict = None) -> Dict[str, Any]:
        """Ingest a document into vector store"""
        pass

    @abstractmethod
    def check_exists(self, document_id: str) -> bool:
        """Check if document already exists"""
        pass


class IProcessService(ABC):
    """Interface for document processing workflows"""

    @abstractmethod
    def process(
        self,
        workflow_id: str,
        document_id: str,
        document_type: str,
        query: str,
        nodes: List[Dict],
        edges: List[Dict],
        start_node: str
    ) -> Dict[str, Any]:
        """Process document using workflow"""
        pass
