"""
Document processing service V2 implementation using LangChain 1.x create_agent
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.services.interfaces import IVectorStore
from app.services.workflow_builder_v2 import process_document_v2


class ProcessServiceV2:
    """
    V2 Process Service using LangChain 1.x create_agent abstraction.

    This is a simplified service that uses the new agent architecture
    instead of manually building LangGraph workflows.
    """

    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store

    def process(
        self,
        workflow_id: str,
        document_id: str,
        document_type: str,
        query: str,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0,
        k_per_query: int = 10,
        max_model_calls: int = 10,
        enable_summarization: bool = False,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process document using LangChain 1.x create_agent.

        Args:
            workflow_id: Workflow identifier for tracking
            document_id: Document to process
            document_type: Type of document for filtering
            query: User query/instruction
            system_prompt: System prompt for the agent
            model: Optional model name override
            temperature: LLM temperature
            k_per_query: Number of results per search query
            max_model_calls: Maximum model calls allowed
            enable_summarization: Enable summarization middleware
            response_format: Optional structured output schema

        Returns:
            Dict with processing results
        """
        print(f"\n{'='*60}")
        print(f"=== PROCESS SERVICE V2 ===")
        print(f"{'='*60}")
        print(f"Workflow ID: {workflow_id}")
        print(f"Document ID: {document_id}")
        print(f"Document Type: {document_type}")
        print(f"Query: {query[:100]}...")
        print(f"Model: {model or 'default'}")
        print(f"Max Model Calls: {max_model_calls}")
        print(f"Summarization: {enable_summarization}")
        print(f"{'='*60}\n")

        try:
            # Process using V2 agent
            result = process_document_v2(
                vector_store=self.vector_store,
                document_id=document_id,
                document_type=document_type,
                query=query,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                k_per_query=k_per_query,
                max_model_calls=max_model_calls,
                enable_summarization=enable_summarization,
                response_format=response_format
            )

            print(f"\n[V2 SERVICE] Processing completed: {result['status']}")

            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": result["status"],
                "message": result["message"],
                "final_output": result["final_output"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            print(f"\n[V2 SERVICE ERROR] {str(e)}")
            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": "failed",
                "message": f"V2 processing failed: {str(e)}",
                "final_output": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
