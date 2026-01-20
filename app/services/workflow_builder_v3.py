"""
LangGraph Workflow Builder V3 - Uses gRPC client for retrieval instead of embedded vector store

This version builds LangGraph workflows (like V1) but uses gRPC calls
for all vector search operations instead of embedded vector store.
"""

from typing import List, Dict, Any, TypedDict, Annotated
from functools import partial
import operator
import json
import time

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks import StdOutCallbackHandler

from app.grpc.retrieval_client import RetrievalClient, RetrievedChunk
from app.core.config import azure_config


# Callback handler for logging LLM interactions
callback_handler = StdOutCallbackHandler()


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
    retrieved_context: str
    current_output: Dict[str, Any]
    node_results: Annotated[List[Dict[str, Any]], operator.add]


# ==================== Node Implementations (gRPC Version) ====================

def create_input_node_grpc(node_config: Dict[str, Any], retrieval_client: RetrievalClient):
    """Creates an input node that retrieves document context via gRPC"""

    def input_node(state: WorkflowState) -> Dict[str, Any]:
        """Retrieves relevant chunks from the vector store via gRPC"""
        try:
            document_id = state["document_id"]
            document_type = state["document_type"]
            query = state["query"]

            print(f"\n[INPUT NODE gRPC] Searching for: '{query[:50]}...'")

            # Search for relevant chunks via gRPC
            chunks = retrieval_client.search(
                query=query,
                document_id=document_id,
                document_type=document_type,
                k=25
            )

            # Combine chunk content
            context_parts = []
            for chunk in chunks:
                context_parts.append(chunk.content)

            context = "\n\n---\n\n".join(context_parts)

            print(f"[INPUT NODE gRPC] Retrieved {len(chunks)} chunks")

            return {
                "retrieved_context": context,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {"chunks_retrieved": len(chunks), "retrieval_mode": "grpc"}
                }]
            }

        except Exception as e:
            print(f"[INPUT NODE gRPC] Error: {e}")
            return {
                "retrieved_context": "",
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "error",
                    "error": str(e)
                }]
            }

    return input_node


def create_retrieval_tool_grpc(
    retrieval_client: RetrievalClient,
    document_id: str,
    document_type: str,
    k: int = 10
):
    """Creates a retrieval tool that uses gRPC client"""

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
        print(f"[gRPC TOOL] Searching: '{query[:50]}...'")

        chunks = retrieval_client.search(
            query=query,
            document_id=document_id,
            document_type=document_type,
            k=k
        )

        if not chunks:
            return "No relevant information found for this query."

        result_parts = []
        for chunk in chunks:
            result_parts.append(f"[Relevance: {chunk.score:.3f}]\n{chunk.content}")

        return "\n\n---\n\n".join(result_parts)

    return search_document


def create_retriever_node_grpc(node_config: Dict[str, Any], retrieval_client: RetrievalClient):
    """
    Creates a retriever node that does retrieval via gRPC.

    Supports two modes via config:
    - use_as_tool: false (default) - Deterministic multi-query retrieval from previous agent's queries
    - use_as_tool: true - Creates an LLM agent with retrieval tool that decides when/what to search
    """

    def retriever_node_deterministic(state: WorkflowState) -> Dict[str, Any]:
        """Deterministic retrieval: Retrieves chunks for multiple queries from previous agent output via gRPC"""
        try:
            document_id = state["document_id"]
            document_type = state["document_type"]
            previous_output = state.get("current_output", {})

            # Get queries from previous output
            # Expects previous agent to output: {"queries": ["query1", "query2", ...]}
            queries = previous_output.get("queries", [])

            if not queries:
                # Fallback to original query if no decomposed queries
                queries = [state["query"]]

            config = node_config.get("config", {}) or {}
            k_per_query = config.get("k_per_query", 10)
            deduplicate = config.get("deduplicate", True)

            print(f"\n[RETRIEVER NODE gRPC] Processing {len(queries)} queries")

            # Use multi_search for efficiency
            chunks = retrieval_client.multi_search(
                queries=queries,
                document_id=document_id,
                document_type=document_type,
                k_per_query=k_per_query,
                deduplicate=deduplicate
            )

            # Group results by query for reporting
            query_results = []
            for query in queries:
                count = len([c for c in chunks if c.query == query])
                query_results.append({
                    "query": query,
                    "chunks_retrieved": count
                })

            # Combine all chunks
            context = "\n\n---\n\n".join([chunk.content for chunk in chunks])

            print(f"[RETRIEVER NODE gRPC] Retrieved {len(chunks)} total chunks")

            return {
                "retrieved_context": context,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {
                        "mode": "deterministic",
                        "retrieval_mode": "grpc",
                        "total_chunks": len(chunks),
                        "queries_processed": len(queries),
                        "query_results": query_results
                    }
                }]
            }

        except Exception as e:
            print(f"[RETRIEVER NODE gRPC] Error: {e}")
            return {
                "retrieved_context": "",
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "error",
                    "error": str(e)
                }]
            }

    def retriever_node_as_tool(state: WorkflowState) -> Dict[str, Any]:
        """Tool-based retrieval: LLM agent with gRPC retrieval tool decides when/what to search"""
        try:
            document_id = state["document_id"]
            document_type = state["document_type"]
            config = node_config.get("config", {}) or {}

            # Create the gRPC retrieval tool
            k_per_query = config.get("k_per_query", 10)
            retrieval_tool = create_retrieval_tool_grpc(
                retrieval_client, document_id, document_type, k=k_per_query
            )

            # Create LLM with tool binding
            llm = AzureChatOpenAI(
                api_key=azure_config.api_key,
                azure_endpoint=azure_config.endpoint,
                api_version=azure_config.api_version,
                model=config.get("model", azure_config.llm_model),
                temperature=config.get("temperature", 0)
            )

            llm_with_tools = llm.bind_tools([retrieval_tool])

            # System prompt for the retrieval agent
            system_prompt = config.get("system_prompt", """You are a document retrieval agent. Your task is to search the document to find all relevant information needed to answer the user's query.

                Use the search_document tool to find relevant information. You should:
                1. Break down complex queries into simpler search queries
                2. Search multiple times if needed to gather comprehensive information
                3. Try different phrasings if initial searches don't return good results
                4. Once you have gathered enough context, summarize what you found

                Be thorough - it's better to search too much than too little.""")

            # Build the query context
            previous_output = state.get("current_output", {})
            user_query = state["query"]

            user_content = f"Query: {user_query}"
            if previous_output:
                user_content = f"Previous processing result:\n{json.dumps(previous_output, indent=2)}\n\n{user_content}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]

            # Agentic loop - let LLM call tools until it's done
            all_retrieved_chunks = []
            tool_calls_made = []
            max_iterations = config.get("max_tool_iterations", 5)

            print(f"\n{'='*60}")
            print(f"[RETRIEVER AGENT gRPC] Starting tool-based retrieval")
            print(f"[RETRIEVER AGENT gRPC] Query: {user_query}")
            print(f"{'='*60}\n")

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
                    # LLM is done searching, extract final response
                    print(f"\n[RETRIEVER AGENT gRPC] LLM finished - no more tool calls")
                    if hasattr(response, 'content') and response.content:
                        print(f"[RETRIEVER AGENT gRPC] Final response: {response.content[:300]}...")
                    break

                # Process each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "search_document":
                        query = tool_args.get("query", "")
                        print(f"\n[TOOL CALL gRPC] search_document")
                        print(f"[TOOL CALL gRPC] Query: '{query}'")

                        tool_result = retrieval_tool.invoke(query)
                        all_retrieved_chunks.append(tool_result)
                        tool_calls_made.append({
                            "iteration": iteration + 1,
                            "query": query,
                            "result_length": len(tool_result)
                        })

                        print(f"[TOOL RESULT gRPC] Retrieved {len(tool_result)} characters")

                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        ))

            # Combine all retrieved content
            context = "\n\n===\n\n".join(all_retrieved_chunks) if all_retrieved_chunks else ""

            # Get the final response content (LLM's summary/analysis)
            final_response = messages[-1].content if hasattr(messages[-1], 'content') else ""

            return {
                "retrieved_context": context,
                "current_output": {
                    "retrieval_summary": final_response,
                    "tool_calls": tool_calls_made
                },
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {
                        "mode": "tool_based",
                        "retrieval_mode": "grpc",
                        "iterations": len(tool_calls_made),
                        "tool_calls": tool_calls_made,
                        "final_summary": final_response[:500] + "..." if len(final_response) > 500 else final_response
                    }
                }]
            }

        except Exception as e:
            print(f"[RETRIEVER AGENT gRPC] Error: {e}")
            return {
                "retrieved_context": "",
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "error",
                    "error": str(e)
                }]
            }

    # Return appropriate node function based on config
    config = node_config.get("config", {}) or {}
    use_as_tool = config.get("use_as_tool", False)

    if use_as_tool:
        return retriever_node_as_tool
    else:
        return retriever_node_deterministic


def create_agent_node_grpc(node_config: Dict[str, Any], retrieval_client: RetrievalClient):
    """Creates an agent node with LLM processing (same as V1, no retrieval needed here)"""

    def agent_node(state: WorkflowState) -> Dict[str, Any]:
        """Processes with LLM, optionally with structured output"""
        try:
            config = node_config.get("config", {}) or {}

            # Create LLM
            llm = AzureChatOpenAI(
                api_key=azure_config.api_key,
                azure_endpoint=azure_config.endpoint,
                api_version=azure_config.api_version,
                model=config.get("model", azure_config.llm_model),
                temperature=config.get("temperature", 0)
            )

            # Get system prompt
            system_prompt = config.get("system_prompt", "You are a helpful assistant.")

            # Check for structured output
            output_format = config.get("output_format")

            # Build the prompt with context
            context = state.get("retrieved_context", "")
            previous_output = state.get("current_output", {})

            user_content = f"""Context from document:
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

            print(f"\n{'='*60}")
            print(f"[AGENT NODE] {node_config['node_name']}")
            print(f"[AGENT NODE] Processing query: {state['query'][:100]}...")
            print(f"{'='*60}\n")

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

            return {
                "current_output": output,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": output
                }]
            }

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


def create_agent_with_tools_node_grpc(node_config: Dict[str, Any], retrieval_client: RetrievalClient):
    """
    Creates an agent node that has retrieval tool available.

    The agent (LLM) can call the retrieval tool when it needs to search the document.
    The retrieval tool itself does NOT use an LLM - it's just vector search via gRPC.
    """

    def agent_with_tools_node(state: WorkflowState) -> Dict[str, Any]:
        """Agent with retrieval tool - LLM decides when to search"""
        try:
            document_id = state["document_id"]
            document_type = state["document_type"]
            config = node_config.get("config", {}) or {}

            # Create the retrieval tool (no LLM, just gRPC search)
            k_per_query = config.get("k_per_query", 10)
            retrieval_tool = create_retrieval_tool_grpc(
                retrieval_client, document_id, document_type, k=k_per_query
            )

            # Create LLM with tool binding
            llm = AzureChatOpenAI(
                api_key=azure_config.api_key,
                azure_endpoint=azure_config.endpoint,
                api_version=azure_config.api_version,
                model=config.get("model", azure_config.llm_model),
                temperature=config.get("temperature", 0)
            )

            llm_with_tools = llm.bind_tools([retrieval_tool])

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

            print(f"\n{'='*60}")
            print(f"[AGENT WITH TOOLS] {node_config['node_name']}")
            print(f"[AGENT WITH TOOLS] Query: {user_query[:100]}...")
            print(f"[AGENT WITH TOOLS] Max iterations: {max_iterations}")
            print(f"{'='*60}\n")

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
                        print(f"[TOOL CALL] search_document (no LLM)")
                        print(f"[TOOL CALL] Query: '{query}'")

                        # Execute tool (no LLM, just gRPC retrieval)
                        tool_result = retrieval_tool.invoke(query)
                        all_retrieved_chunks.append(tool_result)
                        tool_calls_made.append({
                            "iteration": iteration + 1,
                            "query": query,
                            "result_length": len(tool_result)
                        })

                        print(f"[TOOL RESULT] Retrieved {len(tool_result)} characters")

                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        ))

            # Get final response content
            final_content = messages[-1].content if hasattr(messages[-1], 'content') else ""

            # Combine all retrieved context
            combined_context = "\n\n===\n\n".join(all_retrieved_chunks) if all_retrieved_chunks else ""

            # Check for structured output
            output_format = config.get("output_format")
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


# ==================== Workflow Builder (gRPC Version) ====================

def build_workflow_v3(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    retrieval_client: RetrievalClient
):
    """
    Builds a LangGraph workflow from node and edge definitions using gRPC for retrieval.

    Args:
        nodes: List of node definitions with node_id, node_name, node_category, config
        edges: List of edge definitions with source and target
        retrieval_client: gRPC client for document retrieval

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
            node_fn = create_input_node_grpc(node, retrieval_client)
            graph.add_node(node_id, node_fn)

        elif category == "retriever":
            node_fn = create_retriever_node_grpc(node, retrieval_client)
            graph.add_node(node_id, node_fn)

        elif category == "agent":
            node_fn = create_agent_node_grpc(node, retrieval_client)
            graph.add_node(node_id, node_fn)

        elif category == "agent_with_tools":
            node_fn = create_agent_with_tools_node_grpc(node, retrieval_client)
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
        "retrieved_context": "",
        "current_output": {},
        "node_results": []
    }

    result = workflow.invoke(initial_state)

    return result
