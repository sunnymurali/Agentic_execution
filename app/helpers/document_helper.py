"""
Document helper functions
"""

import hashlib
import os
from langchain_pymupdf4llm import PyMuPDF4LLMLoader


def generate_document_id(pdf_path: str) -> str:
    """Generate unique document ID from filename"""
    filename = os.path.basename(pdf_path)
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def get_document_name(pdf_path: str) -> str:
    """Get document name from path"""
    return os.path.basename(pdf_path)


def build_document_path(folder_path: str, document_id: str, extension: str = ".pdf") -> str:
    """Build full document path from folder and document_id"""
    return os.path.join(folder_path, f"{document_id}{extension}")


def read_pdf_text(path: str) -> str:
    """Extract text from PDF using PyMuPDF4LLM"""
    loader = PyMuPDF4LLMLoader(path, mode="page")
    docs = loader.load()
    return "\n\n".join([doc.page_content for doc in docs])


def load_pdf_documents(path: str) -> list:
    """Load PDF as list of LangChain documents"""
    loader = PyMuPDF4LLMLoader(path, mode="page")
    return loader.load()
