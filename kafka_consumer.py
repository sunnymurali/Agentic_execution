"""
Kafka Consumer - Processes workflow events and responds with results
"""

import json
import signal
import sys
from datetime import datetime, timezone
from kafka import KafkaConsumer, KafkaProducer

# Add app to path for imports
sys.path.insert(0, ".")

from app.services.process_service import ProcessService
from app.services.process_service_v2 import ProcessServiceV2
from app.services.vector_store_service import QdrantVectorStoreService
from app.core.vector_store import (
    get_qdrant_client,
    get_embeddings,
    get_vector_store as get_langchain_vector_store,
    ensure_collection_exists
)


def safe_json_deserializer(value):
    """Safely deserialize JSON, return None if invalid"""
    if value is None or len(value) == 0:
        return None
    try:
        return json.loads(value.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: Could not decode message: {e}")
        return None


def create_consumer(
    topic: str,
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "document-processor-group",
    auto_offset_reset: str = "earliest"
) -> KafkaConsumer:
    """Create a Kafka consumer with safe JSON deserialization"""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        auto_commit_interval_ms=5000,
        value_deserializer=safe_json_deserializer,
        consumer_timeout_ms=-1,
    )
    return consumer


def create_producer(bootstrap_servers: str = "localhost:9092") -> KafkaProducer:
    """Create a Kafka producer for sending responses"""
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )
    return producer


def get_vector_store() -> QdrantVectorStoreService:
    """Initialize the vector store"""
    client = get_qdrant_client()
    ensure_collection_exists(client)
    embeddings = get_embeddings()
    langchain_store = get_langchain_vector_store(client, embeddings)
    return QdrantVectorStoreService(client, langchain_store)


def get_process_service(vector_store: QdrantVectorStoreService) -> ProcessService:
    """Initialize the V1 process service"""
    return ProcessService(vector_store)


def get_process_service_v2(vector_store: QdrantVectorStoreService) -> ProcessServiceV2:
    """Initialize the V2 process service (LangChain 1.x create_agent)"""
    return ProcessServiceV2(vector_store)


def process_workflow_message(
    message: dict,
    process_service: ProcessService,
    process_service_v2: ProcessServiceV2
) -> dict:
    """
    Process a workflow message and return results.
    Routes to V1 or V2 processor based on 'processor_type' field.

    processor_type: "v1" (default) or "v2"
    """

    workflow_id = message.get("workflow_id", "unknown")
    document_id = message.get("document_id", "")
    document_type = message.get("document_type", "")
    query = message.get("query", "")
    processor_type = message.get("processor_type", "v1").lower()

    # Common validation
    if not document_id:
        return {
            "workflow_id": workflow_id,
            "document_id": document_id,
            "status": "failed",
            "message": "Missing required field: document_id",
            "final_output": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    if not document_type:
        return {
            "workflow_id": workflow_id,
            "document_id": document_id,
            "status": "failed",
            "message": "Missing required field: document_type",
            "final_output": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    if not query:
        return {
            "workflow_id": workflow_id,
            "document_id": document_id,
            "status": "failed",
            "message": "Missing required field: query",
            "final_output": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # Route based on processor_type
    if processor_type == "v2":
        # V2 processor - uses create_agent, requires system_prompt
        system_prompt = message.get("system_prompt", "")
        if not system_prompt:
            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": "failed",
                "message": "Missing required field for V2: system_prompt",
                "final_output": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        print(f"[ROUTING] Using V2 processor (create_agent)")
        result = process_service_v2.process(
            workflow_id=workflow_id,
            document_id=document_id,
            document_type=document_type,
            query=query,
            system_prompt=system_prompt,
            model=message.get("model"),
            temperature=message.get("temperature", 0),
            k_per_query=message.get("k_per_query", 10),
            max_model_calls=message.get("max_model_calls", 10),
            enable_summarization=message.get("enable_summarization", False),
            response_format=message.get("response_format")
        )
    else:
        # V1 processor - uses LangGraph workflow, requires nodes/edges
        nodes = message.get("nodes", [])
        edges = message.get("edges", [])
        start_node = message.get("start_node", "")

        if not nodes:
            return {
                "workflow_id": workflow_id,
                "document_id": document_id,
                "status": "failed",
                "message": "Missing required field for V1: nodes",
                "final_output": None,
                "node_results": [],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        print(f"[ROUTING] Using V1 processor (LangGraph workflow)")
        result = process_service.process(
            workflow_id=workflow_id,
            document_id=document_id,
            document_type=document_type,
            query=query,
            nodes=nodes,
            edges=edges,
            start_node=start_node
        )

    return result


def run_consumer(
    topic: str,
    response_topic: str = "workflow-responses",
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "document-processor-group"
):
    """Run the Kafka consumer and process workflows"""

    consumer = create_consumer(
        topic=topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id
    )

    producer = create_producer(bootstrap_servers)

    # Initialize vector store (shared by both services)
    print("Initializing vector store...")
    vector_store = get_vector_store()

    # Initialize both process services
    print("Initializing V1 process service (LangGraph workflow)...")
    process_service = get_process_service(vector_store)

    print("Initializing V2 process service (create_agent)...")
    process_service_v2 = get_process_service_v2(vector_store)

    print(f"\nConsumer started")
    print(f"  Bootstrap servers: {bootstrap_servers}")
    print(f"  Listening on: {topic}")
    print(f"  Responding to: {response_topic}")
    print(f"  Supported processor types: v1 (default), v2")
    print(f"\nWaiting for workflow events... (Press Ctrl+C to stop)\n")

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down consumer...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running:
            messages = consumer.poll(timeout_ms=1000)

            for topic_partition, records in messages.items():
                for record in records:
                    if record.value is None:
                        continue

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received workflow:")
                    print(json.dumps(record.value, indent=2))

                    # Process the workflow (routes based on processor_type)
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing...")
                    response = process_workflow_message(
                        record.value,
                        process_service,
                        process_service_v2
                    )

                    # Send response to response topic
                    producer.send(response_topic, value=response)
                    producer.flush()

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Response sent:")
                    print(json.dumps(response, indent=2))

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        consumer.close()
        producer.close()
        print("Consumer closed")


if __name__ == "__main__":
    BOOTSTRAP_SERVERS = "localhost:9092"
    TOPIC = "document-processing"
    RESPONSE_TOPIC = "workflow-responses"
    GROUP_ID = "document-processor-group"

    run_consumer(
        topic=TOPIC,
        response_topic=RESPONSE_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=GROUP_ID
    )
