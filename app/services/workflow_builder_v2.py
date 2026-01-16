"""
LangGraph Workflow Builder V2 - Uses LangChain 1.x create_agent abstraction
"""

from typing import List, Dict, Any, Optional
import json

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.interfaces import IVectorStore
from app.core.config import azure_config


def create_document_search_tool(
    vector_store: IVectorStore,
    document_id: str,
    document_type: str,
    k: int = 10
):
    """
    Creates a document search tool for the agent.
    The tool searches the vector store filtered by document_id and document_type.
    """

    @tool
    def search_document(query: str) -> str:
        """
        Search the document for relevant information based on a query.
        Use this tool to find specific information from the document.
        You can call this tool multiple times with different queries to gather comprehensive information.

        Args:
            query: The search query to find relevant document chunks

        Returns:
            Relevant text chunks from the document
        """
        print(f"\n[V2 TOOL CALL] search_document")
        print(f"[V2 TOOL CALL] Query: '{query}'")

        results = vector_store.search(query, document_id, document_type, k=k)

        if not results:
            print(f"[V2 TOOL RESULT] No results found")
            return "No relevant information found for this query."

        chunks = []
        for doc, score in results:
            chunks.append(f"[Relevance: {score:.3f}]\n{doc.page_content}")

        result = "\n\n---\n\n".join(chunks)
        print(f"[V2 TOOL RESULT] Retrieved {len(result)} characters from {len(results)} chunks")

        return result

    return search_document


def build_agent_v2(
    vector_store: IVectorStore,
    document_id: str,
    document_type: str,
    system_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0,
    k_per_query: int = 10,
    max_model_calls: int = 10,
    enable_summarization: bool = False,
    response_format: Optional[Dict] = None,
    debug: bool = True
):
    """
    Builds an agent using LangChain 1.x create_agent.

    Args:
        vector_store: Vector store for document retrieval
        document_id: ID of the document to search
        document_type: Type of document for filtering
        system_prompt: System prompt for the agent
        model: Model name (defaults to azure_config.llm_model)
        temperature: LLM temperature
        k_per_query: Number of results per search query
        max_model_calls: Maximum model calls allowed
        enable_summarization: Enable summarization middleware
        response_format: Optional structured output format
        debug: Enable debug logging

    Returns:
        Compiled agent graph
    """

    print(f"\n{'='*60}")
    print(f"[V2 AGENT BUILDER] Creating agent")
    print(f"[V2 AGENT BUILDER] Document: {document_id} ({document_type})")
    print(f"[V2 AGENT BUILDER] Model: {model or azure_config.llm_model}")
    print(f"{'='*60}\n")

    # Create the LLM
    llm = AzureChatOpenAI(
        api_key=azure_config.api_key,
        azure_endpoint=azure_config.endpoint,
        api_version=azure_config.api_version,
        model=model or azure_config.llm_model,
        temperature=temperature
    )

    # Create the search tool
    search_tool = create_document_search_tool(
        vector_store=vector_store,
        document_id=document_id,
        document_type=document_type,
        k=k_per_query
    )

    # Build middleware list
    middleware = []

    # Add model call limit middleware
    middleware.append(
        ModelCallLimitMiddleware(
            run_limit=max_model_calls,
            exit_behavior="end"  # Gracefully end when limit reached
        )
    )

    # Optionally add summarization middleware
    if enable_summarization:
        middleware.append(
            SummarizationMiddleware(
                model=llm,
                trigger=("messages", 30),  # Summarize at 30 messages
                keep=("messages", 10)  # Keep most recent 10
            )
        )

    # Create the agent
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=system_prompt,
        #middleware=middleware,
        response_format=response_format,
        debug=debug
    )

    return agent


def execute_agent_v2(
    agent,
    query: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a V2 agent.

    Args:
        agent: Compiled agent from build_agent_v2
        query: User query
        context: Optional additional context

    Returns:
        Dict with agent results
    """

    print(f"\n{'='*60}")
    print(f"[V2 AGENT] Executing agent")
    print(f"[V2 AGENT] Query: {query[:100]}...")
    print(f"{'='*60}\n")

    # Build the input message
    if context:
        user_content = f"""Context:
{context}

Query: {query}"""
    else:
        user_content = query

    # Execute the agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_content}]
    })

    # Extract the final response
    messages = result.get("messages", [])

    # Get the last AI message content
    final_response = None
    tool_calls_made = []

    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # This is an AI message with tool calls
                for tc in msg.tool_calls:
                    tool_calls_made.append({
                        "tool": tc.get("name"),
                        "args": tc.get("args")
                    })
            elif msg.type == "ai":
                final_response = msg.content

    print(f"\n[V2 AGENT] Completed with {len(tool_calls_made)} tool calls")
    if final_response:
        print(f"[V2 AGENT] Final response: {final_response[:200]}...")

    return {
        "final_response": final_response,
        "tool_calls": tool_calls_made,
        "messages": messages,
        "status": "success" if final_response else "no_response"
    }


def process_document_v2(
    vector_store: IVectorStore,
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
    High-level function to process a document using V2 agent.

    This is a simplified interface that builds and executes an agent in one call.

    Args:
        vector_store: Vector store for document retrieval
        document_id: ID of the document to process
        document_type: Type of document for filtering
        query: User query/instruction
        system_prompt: System prompt for the agent
        model: Optional model name
        temperature: LLM temperature
        k_per_query: Number of results per search
        max_model_calls: Maximum model calls
        enable_summarization: Enable summarization middleware
        response_format: Optional structured output format

    Returns:
        Dict with processing results
    """

    try:
        # Build the agent
        agent = build_agent_v2(
            vector_store=vector_store,
            document_id=document_id,
            document_type=document_type,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            k_per_query=k_per_query,
            max_model_calls=max_model_calls,
            enable_summarization=enable_summarization,
            response_format=response_format,
            debug=True
        )

        # Execute the agent
        result = execute_agent_v2(agent, query)

        return {
            "status": result["status"],
            "final_output": {
                "response": result["final_response"],
                "tool_calls": result["tool_calls"]
            },
            "message": "V2 agent executed successfully"
        }

    except Exception as e:
        print(f"[V2 AGENT ERROR] {str(e)}")
        return {
            "status": "error",
            "final_output": None,
            "message": f"V2 agent execution failed: {str(e)}"
        }
