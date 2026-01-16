"""
Document processing service implementation using LangGraph
"""

from typing import Dict, List, Any
from datetime import datetime, timezone

from app.services.interfaces import IProcessService, IVectorStore
from app.services.workflow_builder import build_workflow, execute_workflow


class ProcessService(IProcessService):
    """Concrete implementation of IProcessService using LangGraph"""

    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store

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
        print(f"=== PROCESS SERVICE ===")
        print(f"Workflow ID: {workflow_id}")
        print(f"Document ID: {document_id}")
        print(f"Document Type: {document_type}")
        print(f"Query: {query}")
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(edges)}")
        print(f"Start Node: {start_node}")

        try:
            # Build the workflow from definitions
            print("Building LangGraph workflow...")
            workflow = build_workflow(
                nodes=nodes,
                edges=edges,
                vector_store=self.vector_store
            )

            # Execute the workflow
            print("Executing workflow...")
            result = execute_workflow(
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

            print(f"Workflow {status}: {len(node_results)} nodes executed")

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
            print(f"Workflow execution failed: {e}")
            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": "failed",
                "message": f"Workflow execution failed: {str(e)}",
                "final_output": None,
                "node_results": [],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
