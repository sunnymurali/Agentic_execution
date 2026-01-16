"""
LangGraph Workflow Builder - Dynamically builds workflows from JSON definitions
"""

from typing import List, Dict, Any, TypedDict, Annotated
from functools import partial
import operator
import json

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks import StdOutCallbackHandler

from app.services.interfaces import IVectorStore
from app.core.config import azure_config


# Callback handler for logging LLM interactions
callback_handler = StdOutCallbackHandler()


# ==================== Workflow State ====================

class WorkflowState(TypedDict):
    """State that flows through the workflow"""
    document_id: str
    document_type: str
    query: str
    retrieved_context: str
    current_output: Dict[str, Any]
    node_results: Annotated[List[Dict[str, Any]], operator.add]


# ==================== Node Implementations ====================

def create_input_node(node_config: Dict[str, Any], vector_store: IVectorStore):
    """Creates an input node that retrieves document context"""

    def input_node(state: WorkflowState) -> Dict[str, Any]:
        """Retrieves relevant chunks from the vector store"""
        try:
            document_id = state["document_id"]
            document_type = state["document_type"]
            query = state["query"]

            # Search for relevant chunks filtered by document_id and document_type
            results = vector_store.search(query, document_id, document_type, k=25)

            # Combine chunk content
            context_parts = []
            for doc, score in results:
                context_parts.append(doc.page_content)

            context = "\n\n---\n\n".join(context_parts)

            return {
                "retrieved_context": context,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {"chunks_retrieved": len(results)}
                }]
            }

        except Exception as e:
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


def create_retrieval_tool(vector_store: IVectorStore, document_id: str, document_type: str, k: int = 10):
    """Creates a retrieval tool that can be bound to an LLM"""

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
        results = vector_store.search(query, document_id, document_type, k=k)

        if not results:
            return "No relevant information found for this query."

        chunks = []
        for doc, score in results:
            chunks.append(f"[Relevance: {score:.3f}]\n{doc.page_content}")

        return "\n\n---\n\n".join(chunks)

    return search_document


def create_retriever_node(node_config: Dict[str, Any], vector_store: IVectorStore):
    """
    Creates a retriever node that does retrieval.

    Supports two modes via config:
    - use_as_tool: false (default) - Deterministic multi-query retrieval from previous agent's queries
    - use_as_tool: true - Creates an LLM agent with retrieval tool that decides when/what to search
    """

    def retriever_node_deterministic(state: WorkflowState) -> Dict[str, Any]:
        """Deterministic retrieval: Retrieves chunks for multiple queries from previous agent output"""
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

            # Combine all chunks
            context = "\n\n---\n\n".join(all_chunks)

            return {
                "retrieved_context": context,
                "node_results": [{
                    "node_id": node_config["node_id"],
                    "node_name": node_config["node_name"],
                    "status": "success",
                    "output": {
                        "mode": "deterministic",
                        "total_chunks": len(all_chunks),
                        "queries_processed": len(queries),
                        "query_results": query_results
                    }
                }]
            }

        except Exception as e:
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
        """Tool-based retrieval: LLM agent with retrieval tool decides when/what to search"""
        try:
            document_id = state["document_id"]
            document_type = state["document_type"]
            config = node_config.get("config", {}) or {}

            # Create the retrieval tool
            k_per_query = config.get("k_per_query", 10)
            retrieval_tool = create_retrieval_tool(vector_store, document_id, document_type, k=k_per_query)

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
            print(f"[RETRIEVER AGENT] Starting tool-based retrieval")
            print(f"[RETRIEVER AGENT] Query: {user_query}")
            print(f"{'='*60}\n")

            for iteration in range(max_iterations):
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
                response = llm_with_tools.invoke(messages, config={"callbacks": [callback_handler]})
                messages.append(response)

                # Check if there are tool calls
                if not response.tool_calls:
                    # LLM is done searching, extract final response
                    print(f"\n[RETRIEVER AGENT] LLM finished - no more tool calls")
                    if hasattr(response, 'content') and response.content:
                        print(f"[RETRIEVER AGENT] Final response: {response.content[:300]}...")
                    break

                # Process each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "search_document":
                        query = tool_args.get("query", "")
                        print(f"\n[TOOL CALL] search_document")
                        print(f"[TOOL CALL] Query: '{query}'")

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
                        "iterations": len(tool_calls_made),
                        "tool_calls": tool_calls_made,
                        "final_summary": final_response[:500] + "..." if len(final_response) > 500 else final_response
                    }
                }]
            }

        except Exception as e:
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


def create_agent_node(node_config: Dict[str, Any], vector_store: IVectorStore):
    """Creates an agent node with LLM processing"""

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

            # Invoke LLM with or without structured output
            if output_format:
                llm_structured = llm.with_structured_output(output_format)
                response = llm_structured.invoke(messages, config={"callbacks": [callback_handler]})
                output = response if isinstance(response, dict) else {"result": response}
            else:
                response = llm.invoke(messages, config={"callbacks": [callback_handler]})
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


# ==================== Workflow Builder ====================

def build_workflow(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    vector_store: IVectorStore
):
    """
    Builds a LangGraph workflow from node and edge definitions.

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
            node_fn = create_input_node(node, vector_store)
            graph.add_node(node_id, node_fn)

        elif category == "retriever":
            node_fn = create_retriever_node(node, vector_store)
            graph.add_node(node_id, node_fn)

        elif category == "agent":
            node_fn = create_agent_node(node, vector_store)
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


def execute_workflow(
    workflow,
    document_id: str,
    document_type: str,
    query: str
) -> Dict[str, Any]:
    """
    Executes a compiled workflow.

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
