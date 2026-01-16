"""
Pydantic request/response schemas
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ==================== Ingest ====================

class IngestRequest(BaseModel):
    document_id: str
    folder_path: str
    document_type: str  # e.g., "earnings", "prospectus", "10k", "10q"
    metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    document_id: str
    status: str
    message: str
    chunk_count: Optional[int] = None
    character_count: Optional[int] = None
    already_indexed: Optional[bool] = None
    timestamp: str


# ==================== Process ====================

class NodeConfig(BaseModel):
    """Configuration for a workflow node"""
    model: Optional[str] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    output_format: Optional[Dict[str, Any]] = None  # JSON schema for structured output


class Node(BaseModel):
    """A node in the workflow graph"""
    node_id: str
    node_name: str
    node_category: str  # "input" or "agent"
    config: Optional[NodeConfig] = None


class Edge(BaseModel):
    """An edge connecting two nodes"""
    edge_id: str
    source: str  # "START" for entry point
    target: str  # "END" for finish point


class ProcessRequest(BaseModel):
    """Request to process a document through a workflow"""
    workflow_id: str
    document_id: str
    document_type: str  # e.g., "earnings", "prospectus", "10k", "10q"
    query: str  # The user query/instruction to process
    nodes: List[Node]
    edges: List[Edge]
    start_node: str


class NodeResult(BaseModel):
    """Result from a single node execution"""
    node_id: str
    node_name: str
    status: str  # "success", "error", "skipped"
    output: Optional[Dict[str, Any]] = None  # Structured output from agent
    error: Optional[str] = None


class ProcessResponse(BaseModel):
    """Response from workflow processing"""
    workflow_id: str
    document_id: str
    status: str  # "completed", "failed", "partial"
    message: str
    final_output: Optional[Dict[str, Any]] = None  # Last agent's structured output
    node_results: List[NodeResult]
    timestamp: str


# ==================== Process V2 (LangChain 1.x create_agent) ====================

class ProcessRequestV2(BaseModel):
    """
    Request to process a document using V2 agent (LangChain 1.x create_agent).

    This is a simplified request format that doesn't require defining nodes/edges.
    The agent handles tool calling and reasoning automatically.
    """
    workflow_id: str
    document_id: str
    document_type: str  # e.g., "earnings", "prospectus", "10k", "10q"
    query: str  # The user query/instruction to process
    system_prompt: str  # System prompt for the agent

    # Optional configuration
    model: Optional[str] = None  # Model name override
    temperature: Optional[float] = 0  # LLM temperature
    k_per_query: Optional[int] = 10  # Number of results per search
    max_model_calls: Optional[int] = 10  # Max model calls (prevents infinite loops)
    enable_summarization: Optional[bool] = False  # Enable summarization middleware
    response_format: Optional[Dict[str, Any]] = None  # Structured output schema


class ToolCallResult(BaseModel):
    """Result from a tool call made by the agent"""
    tool: str
    args: Dict[str, Any]


class ProcessResponseV2(BaseModel):
    """Response from V2 workflow processing"""
    workflow_id: str
    document_id: str
    status: str  # "success", "error", "no_response"
    message: str
    final_output: Optional[Dict[str, Any]] = None  # Contains response and tool_calls
    timestamp: str
