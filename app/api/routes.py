"""
FastAPI Routes for Execution Service
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone

from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    ProcessRequest,
    ProcessResponse,
    NodeResult,
    # V2 schemas
    ProcessRequestV2,
    ProcessResponseV2,
    # V3 schemas (gRPC retrieval)
    ProcessRequestV3,
    ProcessResponseV3,
)
from app.services.interfaces import IVectorStore, IIngestService, IProcessService
from app.services.vector_store_service import QdrantVectorStoreService
from app.services.ingest_service import IngestService
from app.services.process_service import ProcessService
from app.services.process_service_v2 import ProcessServiceV2
from app.services.process_service_v3 import ProcessServiceV3
from app.core.vector_store import (
    get_qdrant_client,
    get_embeddings,
    get_vector_store as get_langchain_vector_store,
    ensure_collection_exists
)


router = APIRouter()


# ==================== Dependency Injection ====================

def get_vector_store() -> IVectorStore:
    """Factory for vector store - swap implementations here"""
    client = get_qdrant_client()
    ensure_collection_exists(client)
    embeddings = get_embeddings()
    langchain_store = get_langchain_vector_store(client, embeddings)
    return QdrantVectorStoreService(client, langchain_store)


def get_ingest_service() -> IIngestService:
    """Factory for ingest service"""
    vector_store = get_vector_store()
    return IngestService(vector_store)


def get_process_service() -> IProcessService:
    """Factory for process service"""
    vector_store = get_vector_store()
    return ProcessService(vector_store)


def get_process_service_v2() -> ProcessServiceV2:
    """Factory for V2 process service (LangChain 1.x create_agent)"""
    vector_store = get_vector_store()
    return ProcessServiceV2(vector_store)


def get_process_service_v3(
    retrieval_host: str = "localhost",
    retrieval_port: int = 50051
) -> ProcessServiceV3:
    """Factory for V3 process service (gRPC retrieval)"""
    return ProcessServiceV3(
        retrieval_host=retrieval_host,
        retrieval_port=retrieval_port
    )


# ==================== Endpoints ====================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the vector store.

    - Loads PDF from folder_path using document_id as filename
    - Chunks the document using RecursiveCharacterTextSplitter
    - Generates embeddings using Azure OpenAI
    - Stores in Qdrant vector database
    """
    try:
        ingest_service = get_ingest_service()

        result = ingest_service.ingest(
            document_id=request.document_id,
            folder_path=request.folder_path,
            document_type=request.document_type,
            metadata=request.metadata
        )

        return IngestResponse(
            document_id=result["document_id"],
            status=result["status"],
            message=result["message"],
            chunk_count=result.get("chunk_count"),
            character_count=result.get("character_count"),
            already_indexed=result.get("already_indexed"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@router.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Process a document using a workflow definition.

    - Builds a LangGraph workflow from nodes and edges
    - Executes the workflow starting from start_node
    - Returns status for each node execution with structured output
    """
    try:
        process_service = get_process_service()

        # Convert Pydantic models to dicts for the service
        nodes_dict = [node.model_dump() for node in request.nodes]
        edges_dict = [edge.model_dump() for edge in request.edges]

        result = process_service.process(
            workflow_id=request.workflow_id,
            document_id=request.document_id,
            document_type=request.document_type,
            query=request.query,
            nodes=nodes_dict,
            edges=edges_dict,
            start_node=request.start_node
        )

        node_results = [
            NodeResult(
                node_id=nr["node_id"],
                node_name=nr["node_name"],
                status=nr["status"],
                output=nr.get("output"),
                error=nr.get("error")
            )
            for nr in result.get("node_results", [])
        ]

        return ProcessResponse(
            workflow_id=result["workflow_id"],
            document_id=result["document_id"],
            status=result["status"],
            message=result["message"],
            final_output=result.get("final_output"),
            node_results=node_results,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Process failed: {str(e)}")


@router.post("/process_v2", response_model=ProcessResponseV2)
async def process_document_v2(request: ProcessRequestV2):
    """
    Process a document using V2 agent (LangChain 1.x create_agent).

    This is a simplified endpoint that uses the new agent architecture:
    - No need to define nodes/edges - agent handles tool calling automatically
    - Built-in middleware support (summarization, model call limits)
    - Simpler request format with just query and system_prompt

    The agent will:
    1. Receive the query
    2. Automatically search the document using the search_document tool
    3. Reason about the results and make additional searches if needed
    4. Return the final response
    """
    try:
        process_service = get_process_service_v2()

        result = process_service.process(
            workflow_id=request.workflow_id,
            document_id=request.document_id,
            document_type=request.document_type,
            query=request.query,
            system_prompt=request.system_prompt,
            model=request.model,
            temperature=request.temperature,
            k_per_query=request.k_per_query,
            max_model_calls=request.max_model_calls,
            enable_summarization=request.enable_summarization,
            response_format=request.response_format
        )

        return ProcessResponseV2(
            workflow_id=result["workflow_id"],
            document_id=result["document_id"],
            status=result["status"],
            message=result["message"],
            final_output=result.get("final_output"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Process V2 failed: {str(e)}")


@router.post("/process_v3", response_model=ProcessResponseV3)
async def process_document_v3(request: ProcessRequestV3):
    """
    Process a document using V3 (LangGraph with gRPC retrieval).

    Same workflow structure as V1 (nodes, edges) but uses gRPC for retrieval:
    - Builds a LangGraph workflow from nodes and edges
    - All retrieval calls go through gRPC to a separate retrieval service
    - Enables independent scaling of retrieval and processing

    Requirements:
    - gRPC retrieval server must be running (default: localhost:50051)
    - Start the server with: python -m app.grpc.retrieval_server

    The workflow will:
    1. Build LangGraph workflow from nodes and edges
    2. Execute nodes using gRPC retrieval service for document search
    3. Return status for each node execution with structured output
    """
    try:
        process_service = get_process_service_v3(
            retrieval_host=request.retrieval_host,
            retrieval_port=request.retrieval_port
        )

        # Convert Pydantic models to dicts for the service
        nodes_dict = [node.model_dump() for node in request.nodes]
        edges_dict = [edge.model_dump() for edge in request.edges]

        result = process_service.process(
            workflow_id=request.workflow_id,
            document_id=request.document_id,
            document_type=request.document_type,
            query=request.query,
            nodes=nodes_dict,
            edges=edges_dict,
            start_node=request.start_node
        )

        node_results = [
            NodeResult(
                node_id=nr["node_id"],
                node_name=nr["node_name"],
                status=nr["status"],
                output=nr.get("output"),
                error=nr.get("error")
            )
            for nr in result.get("node_results", [])
        ]

        return ProcessResponseV3(
            workflow_id=result["workflow_id"],
            document_id=result["document_id"],
            status=result["status"],
            message=result["message"],
            final_output=result.get("final_output"),
            node_results=node_results,
            retrieval_mode=result.get("retrieval_mode", "grpc"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Process V3 failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
