"""
Process Service V3 - Uses LangGraph with gRPC for retrieval

This service uses LangGraph workflows (like V1) but decouples retrieval
by calling a gRPC retrieval service instead of using embedded vector store.
"""

from typing import Dict, List, Any
from datetime import datetime, timezone

from app.services.workflow_builder_v3 import build_workflow_v3, execute_workflow_v3
from app.grpc.retrieval_client import get_retrieval_client


class ProcessServiceV3:
    """
    Process service that uses LangGraph with gRPC retrieval.

    Same workflow structure as V1 (nodes, edges) but uses gRPC
    for all vector search operations instead of embedded vector store.
    """

    def __init__(
        self,
        retrieval_host: str = "localhost",
        retrieval_port: int = 50051
    ):
        """
        Initialize the V3 process service.

        Args:
            retrieval_host: gRPC retrieval service host
            retrieval_port: gRPC retrieval service port
        """
        self.retrieval_host = retrieval_host
        self.retrieval_port = retrieval_port
        self.retrieval_client = None
        print(f"[ProcessServiceV3] Initialized with gRPC retrieval at {retrieval_host}:{retrieval_port}")

    def _get_retrieval_client(self):
        """Get or create retrieval client"""
        if self.retrieval_client is None:
            self.retrieval_client = get_retrieval_client(
                host=self.retrieval_host,
                port=self.retrieval_port
            )
        return self.retrieval_client

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
        Process document using LangGraph workflow with gRPC retrieval.

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
        print(f"[ProcessServiceV3] === PROCESS SERVICE V3 (gRPC) ===")
        print(f"[ProcessServiceV3] Workflow ID: {workflow_id}")
        print(f"[ProcessServiceV3] Document ID: {document_id}")
        print(f"[ProcessServiceV3] Document Type: {document_type}")
        print(f"[ProcessServiceV3] Query: {query[:100]}...")
        print(f"[ProcessServiceV3] Nodes: {len(nodes)}")
        print(f"[ProcessServiceV3] Edges: {len(edges)}")
        print(f"[ProcessServiceV3] Start Node: {start_node}")
        print(f"[ProcessServiceV3] Retrieval via gRPC: {self.retrieval_host}:{self.retrieval_port}")
        print(f"{'='*60}\n")

        try:
            # Get retrieval client
            retrieval_client = self._get_retrieval_client()

            # Build the workflow from definitions using gRPC retrieval
            print("[ProcessServiceV3] Building LangGraph workflow with gRPC retrieval...")
            workflow = build_workflow_v3(
                nodes=nodes,
                edges=edges,
                retrieval_client=retrieval_client
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
            message = "Workflow completed with errors" if has_errors else "Workflow executed successfully via gRPC retrieval"

            print(f"[ProcessServiceV3] Workflow {status}: {len(node_results)} nodes executed")

            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": status,
                "message": message,
                "final_output": final_output,
                "node_results": node_results,
                "retrieval_mode": "grpc",
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
                "retrieval_mode": "grpc",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
