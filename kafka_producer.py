"""
Kafka Producer - Sends JSON events and listens for responses
"""

import json
import signal
import threading
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer


def create_producer(bootstrap_servers: str = "localhost:9092") -> KafkaProducer:
    """Create a Kafka producer with JSON serialization"""
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
    )
    return producer


def send_event(producer: KafkaProducer, topic: str, event: dict, key: str = None):
    """Send a JSON event to Kafka topic"""
    future = producer.send(topic, value=event, key=key)
    record_metadata = future.get(timeout=10)
    print(f"Sent to partition {record_metadata.partition} offset {record_metadata.offset}")
    return record_metadata


def response_listener(bootstrap_servers: str, response_topic: str, running_flag: list):
    """Listen for responses in a separate thread"""
    consumer = KafkaConsumer(
        response_topic,
        bootstrap_servers=bootstrap_servers,
        group_id="producer-response-listener",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v else None,
        consumer_timeout_ms=1000,
    )

    print(f"Listening for responses on: {response_topic}\n")

    while running_flag[0]:
        messages = consumer.poll(timeout_ms=1000)
        for topic_partition, records in messages.items():
            for record in records:
                if record.value:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Response received:")
                    print(json.dumps(record.value, indent=2))
                    print("\n> ", end="", flush=True)

    consumer.close()


def run_producer(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "document-processing",
    response_topic: str = "workflow-responses"
):
    """Run the producer continuously - accepts JSON input and listens for responses"""

    producer = create_producer(bootstrap_servers)
    print(f"Producer connected to {bootstrap_servers}")
    print(f"Sending to: {topic}")
    print(f"Listening for responses on: {response_topic}")
    print(f"\nEnter JSON input (or 'exit' to quit):\n")

    running = [True]

    # Start response listener in a separate thread
    listener_thread = threading.Thread(
        target=response_listener,
        args=(bootstrap_servers, response_topic, running),
        daemon=True
    )
    listener_thread.start()

    def signal_handler(sig, frame):
        running[0] = False
        print("\nShutting down producer...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running[0]:
            user_input = input("> ").strip()

            if user_input.lower() == "exit":
                print("Exiting...")
                break

            if not user_input:
                continue

            try:
                event = json.loads(user_input)
                send_event(producer, topic, event)
                print("Event sent! Waiting for response...\n")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}\n")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        running[0] = False
        producer.flush()
        producer.close()
        print("Producer closed")


if __name__ == "__main__":
    BOOTSTRAP_SERVERS = "localhost:9092"
    TOPIC = "document-processing"
    RESPONSE_TOPIC = "workflow-responses"
    run_producer(BOOTSTRAP_SERVERS, TOPIC, RESPONSE_TOPIC)
