"""
LangGraph Workflow Builder V3 - Vector Search as a LangGraph Node

This version makes vector search a first-class LangGraph node.
Vector search is performed directly against the vector store (no gRPC).

Node Types:
- input: Initializes workflow state (no retrieval)
- vector_search: Performs vector search directly against vector store
- agent: LLM processing using context from state
- agent_with_tools: LLM with vector search as a callable tool (agentic loop)
"""

from typing import List, Dict, Any, TypedDict, Annotated, Optional
from functools import partial
import operator
import json
import time
import re

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks import StdOutCallbackHandler

from app.services.interfaces import IVectorStore
from app.core.config import azure_config


# Callback handler for logging LLM interactions
callback_handler = StdOutCallbackHandler()


def log_node_input(node_name: str, node_id: str, state: dict):
    """Log the input state for a node"""
    print(f"\n{'#'*70}")
    print(f"# NODE INPUT: {node_name} ({node_id})")
    print(f"{'#'*70}")
    print(f"  document_id: {state.get('document_id', 'N/A')}")
    print(f"  document_type: {state.get('document_type', 'N/A')}")
    print(f"  query: {state.get('query', 'N/A')[:100]}..." if state.get('query') else "  query: N/A")

    # Log queries list
    queries = state.get('queries', [])
    print(f"  queries ({len(queries)} total):")
    for i, q in enumerate(queries[:5]):  # Show first 5 queries
        print(f"    [{i}]: {q[:80]}..." if len(q) > 80 else f"    [{i}]: {q}")
    if len(queries) > 5:
        print(f"    ... and {len(queries) - 5} more")

    # Log retrieved context (truncated)
    context = state.get('retrieved_context', '')
    context_len = len(context)
    print(f"  retrieved_context: {context_len} chars")
    if context:
        print(f"    Preview: {context[:200]}..." if len(context) > 200 else f"    Preview: {context}")

    # Log current output
    current_output = state.get('current_output', {})
    if current_output:
        output_str = json.dumps(current_output, indent=2)
        print(f"  current_output:")
        for line in output_str.split('\n')[:10]:  # Show first 10 lines
            print(f"    {line}")
        if output_str.count('\n') > 10:
            print(f"    ... (truncated)")
    else:
        print(f"  current_output: {{}}")

    # Log node results count
    node_results = state.get('node_results', [])
    print(f"  node_results: {len(node_results)} previous results")
    print(f"{'#'*70}\n")


# ==================== Retry Helper for Rate Limits ====================

def invoke_with_retry(
    llm_or_chain,
    messages,
    max_retries: int = 3,
    base_delay: float = 10.0,
    config: dict = None
):
    """
    Invoke LLM with retry logic for 429 rate limit errors.

    Args:
        llm_or_chain: The LLM or chain to invoke
        messages: Messages to pass to invoke
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be multiplied by 2^attempt)
        config: Optional config dict for invoke

    Returns:
        Response from successful invocation

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if config:
                return llm_or_chain.invoke(messages, config=config)
            else:
                return llm_or_chain.invoke(messages)
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error (429)
            if "429" in error_str or "rate limit" in error_str.lower():
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                    print(f"[RATE LIMIT] 429 error on attempt {attempt + 1}/{max_retries + 1}")
                    print(f"[RATE LIMIT] Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                else:
                    print(f"[RATE LIMIT] All {max_retries + 1} attempts failed")
                    raise
            else:
                # Not a rate limit error, raise immediately
                raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


# ==================== Workflow State ====================

class WorkflowState(TypedDict):
    """State that flows through the workflow"""
    document_id: str
    document_type: str
    query: str
    queries: List[str]  # Multiple queries for retrieval (from query decomposition)
    retrieved_context: str
    current_output: Dict[str, Any]
    node_results: Annotated[List[Dict[str, Any]], operator.add]


# ==================== Node Implementations ====================

def create_input_node(node_config: Dict[str, Any]):
    """
    Creates an input node that initializes workflow state.

    This node does NOT perform retrieval - it just sets up the initial state.
    Use a vector_search node after this for retrieval.
    """

    def input_node(state: WorkflowState) -> Dict[str, Any]:
        """Initialize workflow state"""
        try:
            # Log input state
            log_node_input(node_config["node_name"], node_config["node_id"], state)

            print(f"[INPUT NODE] Initializing workflow")

            # Initialize queries list with the main query
            queries = [state["query"]]

            return {
                "queries": queries,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {"initialized": True}
                }]
            }

        except Exception as e:
            print(f"[INPUT NODE] Error: {e}")
            return {
                "queries": [],
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "error",
                    "error": str(e)
                }]
            }

    return input_node


def create_vector_search_node(node_config: Dict[str, Any], vector_store: IVectorStore):
    """
    Creates a vector search node that retrieves documents from the vector store.

    This is the ONLY node that communicates with the vector database.
    It reads queries from state and writes retrieved context to state.

    Config options:
    - k_per_query: Number of results per query (default: 10)
    - deduplicate: Whether to deduplicate results (default: True)
    """

    def vector_search_node(state: WorkflowState) -> Dict[str, Any]:
        """Perform vector search against vector store"""
        try:
            # Log input state
            log_node_input(node_config["node_name"], node_config["node_id"], state)

            document_id = state["document_id"]
            document_type = state["document_type"]
            config = node_config.get("config", {}) or {}

            # Get queries from state
            queries = state.get("queries", [])
            if not queries:
                queries = [state["query"]]

            k_per_query = config.get("k_per_query", 10)
            deduplicate = config.get("deduplicate", True)

            print(f"[VECTOR SEARCH NODE] Processing {len(queries)} queries with k={k_per_query}")

            # Retrieve for each query
            all_chunks = []
            seen_content = set()
            query_results = []

            for query in queries:
                results = vector_store.search(query, document_id, document_type, k=k_per_query)

                chunks_for_query = []
                for doc, score in results:
                    content = doc.page_content
                    # Deduplicate if enabled
                    if deduplicate:
                        content_hash = hash(content[:200])  # Hash first 200 chars
                        if content_hash in seen_content:
                            continue
                        seen_content.add(content_hash)

                    all_chunks.append(content)
                    chunks_for_query.append({"content": content[:200] + "...", "score": score})

                query_results.append({
                    "query": query,
                    "chunks_retrieved": len(chunks_for_query)
                })

            # Combine chunks into context
            context = "\n\n---\n\n".join(all_chunks)

            print(f"[VECTOR SEARCH NODE] Retrieved {len(all_chunks)} total chunks")

            return {
                "retrieved_context": context,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {
                        "total_chunks": len(all_chunks),
                        "queries_processed": len(queries),
                        "query_results": query_results
                    }
                }]
            }

        except Exception as e:
            print(f"[VECTOR SEARCH NODE] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "retrieved_context": "",
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "error",
                    "error": str(e)
                }]
            }

    return vector_search_node


def create_agent_node(node_config: Dict[str, Any]):
    """
    Creates an agent node with LLM processing.

    This node does NOT perform retrieval - it uses context from state.

    Config options:
    - model: LLM model to use
    - temperature: LLM temperature
    - system_prompt: System prompt for the LLM
    - output_format: Optional structured output schema
    - output_queries: If true, outputs {"queries": [...]} for next retrieval node
    """

    def agent_node(state: WorkflowState) -> Dict[str, Any]:
        """Process with LLM using context from state"""
        try:
            # Log input state
            log_node_input(node_config["node_name"], node_config["node_id"], state)

            config = node_config.get("config", {}) or {}

            # Create LLM - use 'or' to handle None values in config
            model_name = config.get("model") or azure_config.llm_model
            print(f"[AGENT NODE] Using model: {model_name}")

            llm = AzureChatOpenAI(
                api_key=azure_config.api_key,
                azure_endpoint=azure_config.endpoint,
                api_version=azure_config.api_version,
                azure_deployment=model_name,
                model=model_name,
                temperature=config.get("temperature", 0)
            )

            # Get system prompt
            system_prompt = config.get("system_prompt", "You are a helpful assistant.")

            # Check for structured output - ensure schema has required title/description
            output_format = config.get("output_format")
            if output_format and isinstance(output_format, dict):
                # LangChain requires 'title' and 'description' at top level for JSON schemas
                # Title must match pattern ^[a-zA-Z0-9_-]+$ (no spaces or special chars)
                if "title" not in output_format:
                    raw_name = node_config.get("node_name", "StructuredOutput")
                    # Sanitize: replace spaces with underscores, remove invalid chars
                    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
                    output_format["title"] = sanitized_name
                if "description" not in output_format:
                    output_format["description"] = f"Structured output for {node_config.get('node_name', 'agent')}"

            # Build the prompt with context
            context = state.get("retrieved_context", "")
            previous_output = state.get("current_output", {})

            user_content = ""
            if context:
                user_content += f"""Context from document:
                                {context}

                                """
            if previous_output:
                user_content += f"""Previous processing result:
                                {json.dumps(previous_output, indent=2)}

                                """
            user_content += f"""Query: {state["query"]}"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]

            print(f"[AGENT NODE] Invoking LLM...")

            # Invoke LLM with or without structured output (with retry for rate limits)
            if output_format:
                llm_structured = llm.with_structured_output(output_format)
                response = invoke_with_retry(
                    llm_structured,
                    messages,
                    max_retries=3,
                    base_delay=10.0,
                    config={"callbacks": [callback_handler]}
                )
                output = response if isinstance(response, dict) else {"result": response}
            else:
                response = invoke_with_retry(
                    llm,
                    messages,
                    max_retries=3,
                    base_delay=10.0,
                    config={"callbacks": [callback_handler]}
                )
                output = {"result": response.content}

            print(f"\n[AGENT NODE] Output: {json.dumps(output, indent=2)[:500]}...")

            # Check if this agent outputs queries for next retrieval
            result = {
                "current_output": output,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": output
                }]
            }

            # If output contains queries, update state for next vector_search node
            if "queries" in output:
                result["queries"] = output["queries"]

            return result

        except Exception as e:
            print(f"[AGENT NODE] Error: {e}")
            return {
                "current_output": {},
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "error",
                    "error": str(e)
                }]
            }

    return agent_node


def create_agent_with_tools_node(node_config: Dict[str, Any], vector_store: IVectorStore):
    """
    Creates an agent node that has vector search as a callable tool.

    This is an AGENTIC node - the LLM decides when to call the search tool.
    The tool itself does NOT use an LLM - it's just vector search.

    This node is self-contained with its own agentic loop:
    1. LLM receives query
    2. LLM decides to call search tool (or not)
    3. Tool executes vector search
    4. LLM receives results and decides next action
    5. Repeat until LLM is done

    Config options:
    - model: LLM model to use
    - temperature: LLM temperature
    - system_prompt: System prompt for the LLM
    - k_per_query: Number of results per tool call (default: 10)
    - max_tool_iterations: Max number of tool call iterations (default: 5)
    - output_format: Optional structured output schema
    """

    def agent_with_tools_node(state: WorkflowState) -> Dict[str, Any]:
        """Agent with vector search tool - LLM decides when to search"""
        try:
            # Log input state
            log_node_input(node_config["node_name"], node_config["node_id"], state)

            document_id = state["document_id"]
            document_type = state["document_type"]
            config = node_config.get("config", {}) or {}

            # Create the vector search tool (no LLM, just vector search)
            k_per_query = config.get("k_per_query", 10)

            @tool
            def search_document(query: str) -> str:
                """
                Search the document for relevant information based on a query.
                Use this tool to find specific information from the document.
                You can call this tool multiple times with different queries.

                Args:
                    query: The search query to find relevant document chunks

                Returns:
                    Relevant text chunks from the document
                """
                print(f"[TOOL: search_document] Query: '{query[:50]}...'")

                results = vector_store.search(query, document_id, document_type, k=k_per_query)

                if not results:
                    return "No relevant information found for this query."

                result_parts = []
                for doc, score in results:
                    result_parts.append(f"[Relevance: {score:.3f}]\n{doc.page_content}")

                result = "\n\n---\n\n".join(result_parts)
                print(f"[TOOL: search_document] Retrieved {len(results)} chunks ({len(result)} chars)")
                return result

            # Create LLM with tool binding - use 'or' to handle None values in config
            model_name = config.get("model") or azure_config.llm_model
            print(f"[AGENT WITH TOOLS] Using model: {model_name}")

            llm = AzureChatOpenAI(
                api_key=azure_config.api_key,
                azure_endpoint=azure_config.endpoint,
                api_version=azure_config.api_version,
                azure_deployment=model_name,
                model=model_name,
                temperature=config.get("temperature", 0)
            )

            llm_with_tools = llm.bind_tools([search_document])

            # Get system prompt
            system_prompt = config.get("system_prompt", """You are a helpful assistant that can search documents.

Use the search_document tool to find relevant information from the document.
You can search multiple times with different queries to gather comprehensive information.
Once you have enough information, provide your final answer.""")

            # Build the query context
            previous_output = state.get("current_output", {})
            context = state.get("retrieved_context", "")
            user_query = state["query"]

            user_content = f"Query: {user_query}"
            if context:
                user_content = f"Previously retrieved context:\n{context[:2000]}...\n\n{user_content}"
            if previous_output:
                user_content = f"Previous processing result:\n{json.dumps(previous_output, indent=2)}\n\n{user_content}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]

            # Agentic loop - LLM calls tools until done
            all_retrieved_chunks = []
            tool_calls_made = []
            max_iterations = config.get("max_tool_iterations", 5)

            print(f"[AGENT WITH TOOLS] Starting agentic loop (max {max_iterations} iterations)")

            for iteration in range(max_iterations):
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
                response = invoke_with_retry(
                    llm_with_tools,
                    messages,
                    max_retries=3,
                    base_delay=10.0,
                    config={"callbacks": [callback_handler]}
                )
                messages.append(response)

                # Check if there are tool calls
                if not response.tool_calls:
                    print(f"[AGENT WITH TOOLS] LLM finished - no more tool calls")
                    break

                # Process each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "search_document":
                        query = tool_args.get("query", "")

                        # Execute tool (no LLM, just vector search)
                        tool_result = search_document.invoke(query)
                        all_retrieved_chunks.append(tool_result)
                        tool_calls_made.append({
                            "iteration": iteration + 1,
                            "query": query,
                            "result_length": len(tool_result)
                        })

                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        ))

            # Get final response content
            final_content = messages[-1].content if hasattr(messages[-1], 'content') else ""

            # Combine all retrieved context
            combined_context = "\n\n===\n\n".join(all_retrieved_chunks) if all_retrieved_chunks else ""

            # Check for structured output - ensure schema has required title/description
            output_format = config.get("output_format")
            if output_format and isinstance(output_format, dict):
                # Title must match pattern ^[a-zA-Z0-9_-]+$ (no spaces or special chars)
                if "title" not in output_format:
                    raw_name = node_config.get("node_name", "StructuredOutput")
                    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
                    output_format["title"] = sanitized_name
                if "description" not in output_format:
                    output_format["description"] = f"Structured output for {node_config.get('node_name', 'agent')}"

            if output_format and final_content:
                print(f"[AGENT WITH TOOLS] Generating structured output...")
                llm_structured = llm.with_structured_output(output_format)
                structured_response = invoke_with_retry(
                    llm_structured,
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Based on this analysis:\n\n{final_content}\n\nProvide the structured output.")
                    ],
                    max_retries=3,
                    base_delay=10.0,
                    config={"callbacks": [callback_handler]}
                )
                output = structured_response if isinstance(structured_response, dict) else {"result": structured_response}
            else:
                output = {"result": final_content}

            print(f"\n[AGENT WITH TOOLS] Completed with {len(tool_calls_made)} tool calls")

            return {
                "retrieved_context": combined_context,
                "current_output": output,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {
                        **output,
                        "tool_calls": tool_calls_made,
                        "iterations": len(tool_calls_made)
                    }
                }]
            }

        except Exception as e:
            print(f"[AGENT WITH TOOLS] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "current_output": {},
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "error",
                    "error": str(e)
                }]
            }

    return agent_with_tools_node


# ==================== Workflow Builder ====================

def build_workflow_v3(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    vector_store: IVectorStore
):
    """
    Builds a LangGraph workflow from node and edge definitions.

    Node categories:
    - input: Initialize workflow state
    - vector_search: Perform vector search (THE retrieval node)
    - agent: LLM processing (no retrieval)
    - agent_with_tools: LLM with vector search as a callable tool

    Args:
        nodes: List of node definitions with node_id, node_name, node_category, config
        edges: List of edge definitions with source and target
        vector_store: Vector store for document retrieval

    Returns:
        Compiled LangGraph workflow
    """

    # Create the state graph
    graph = StateGraph(WorkflowState)

    # Build node lookup
    node_lookup = {node["node_id"]: node for node in nodes}

    # Add nodes based on category
    for node in nodes:
        node_id = node["node_id"]
        category = node["node_category"]

        if category == "input":
            node_fn = create_input_node(node)
            graph.add_node(node_id, node_fn)

        elif category == "vector_search":
            node_fn = create_vector_search_node(node, vector_store)
            graph.add_node(node_id, node_fn)

        elif category == "agent":
            node_fn = create_agent_node(node)
            graph.add_node(node_id, node_fn)

        elif category == "agent_with_tools":
            node_fn = create_agent_with_tools_node(node, vector_store)
            graph.add_node(node_id, node_fn)

        # Legacy support: treat "retriever" as "vector_search"
        elif category == "retriever":
            node_fn = create_vector_search_node(node, vector_store)
            graph.add_node(node_id, node_fn)

        else:
            raise ValueError(f"Unknown node category: {category}")

    # Add edges
    for edge in edges:
        source = edge["source"]
        target = edge["target"]

        if source == "START":
            graph.set_entry_point(target)
        elif target == "END":
            graph.set_finish_point(source)
        else:
            graph.add_edge(source, target)

    # Compile and return
    return graph.compile()


def execute_workflow_v3(
    workflow,
    document_id: str,
    document_type: str,
    query: str
) -> Dict[str, Any]:
    """
    Executes a compiled V3 workflow.

    Args:
        workflow: Compiled LangGraph workflow
        document_id: ID of the document to process
        document_type: Type of document for filtering
        query: User query/instruction

    Returns:
        Final workflow state with results
    """

    initial_state: WorkflowState = {
        "document_id": document_id,
        "document_type": document_type,
        "query": query,
        "queries": [query],  # Initialize with main query
        "retrieved_context": "",
        "current_output": {},
        "node_results": []
    }

    result = workflow.invoke(initial_state)

    return result
