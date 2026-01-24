"""
Process Service V3 - Uses LangGraph with Vector Search as a Node

This service uses LangGraph workflows where vector search is a first-class
node in the graph. Vector search is performed directly against the vector store.
"""

from typing import Dict, List, Any
from datetime import datetime, timezone

from app.services.workflow_builder_v3 import build_workflow_v3, execute_workflow_v3
from app.services.interfaces import IVectorStore


class ProcessServiceV3:
    """
    Process service that uses LangGraph with vector search as a node.

    Node types available:
    - input: Initialize workflow state
    - vector_search: Perform vector search directly against vector store
    - agent: LLM processing using context from state
    - agent_with_tools: LLM with vector search as a callable tool
    """

    def __init__(self, vector_store: IVectorStore):
        """
        Initialize the V3 process service.

        Args:
            vector_store: Vector store for document retrieval
        """
        self.vector_store = vector_store
        print(f"[ProcessServiceV3] Initialized with direct vector store access")

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
        """
        Process document using LangGraph workflow.

        Args:
            workflow_id: Workflow identifier
            document_id: Document to process
            document_type: Type of document for filtering (e.g., earnings, prospectus, 10k, 10q)
            query: User query/instruction
            nodes: Workflow node definitions
            edges: Workflow edge definitions
            start_node: Entry point node ID

        Returns:
            Dict with processing results including final_output and node_results
        """
        print(f"\n{'='*60}")
        print(f"[ProcessServiceV3] === PROCESS SERVICE V3 ===")
        print(f"[ProcessServiceV3] Workflow ID: {workflow_id}")
        print(f"[ProcessServiceV3] Document ID: {document_id}")
        print(f"[ProcessServiceV3] Document Type: {document_type}")
        print(f"[ProcessServiceV3] Query: {query[:100]}...")
        print(f"[ProcessServiceV3] Nodes: {len(nodes)}")
        print(f"[ProcessServiceV3] Edges: {len(edges)}")
        print(f"[ProcessServiceV3] Start Node: {start_node}")
        print(f"{'='*60}\n")

        try:
            # Build the workflow from definitions
            print("[ProcessServiceV3] Building LangGraph workflow...")
            workflow = build_workflow_v3(
                nodes=nodes,
                edges=edges,
                vector_store=self.vector_store
            )

            # Execute the workflow
            print("[ProcessServiceV3] Executing workflow...")
            result = execute_workflow_v3(
                workflow=workflow,
                document_id=document_id,
                document_type=document_type,
                query=query
            )

            # Extract results
            node_results = result.get("node_results", [])
            final_output = result.get("current_output", {})

            # Determine overall status
            has_errors = any(nr.get("status") == "error" for nr in node_results)
            status = "failed" if has_errors else "completed"
            message = "Workflow completed with errors" if has_errors else "Workflow executed successfully"

            print(f"[ProcessServiceV3] Workflow {status}: {len(node_results)} nodes executed")

            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": status,
                "message": message,
                "final_output": final_output,
                "node_results": node_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            print(f"[ProcessServiceV3] Workflow execution failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": "failed",
                "message": f"Workflow execution failed: {str(e)}",
                "final_output": None,
                "node_results": [],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
