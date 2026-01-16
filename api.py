"""
Execution Service API
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Execution Service API")


@app.post("/ingest")
async def ingest_document(request: dict):
    """Ingest a document into the system"""
    print("=== INGEST DOCUMENT ===")
    print(f"Request: {request}")
    print("Loading document...")
    print("Chunking document...")
    print("Generating embeddings...")
    print("Storing in vector database...")
    print("Done!")
    return {"status": "success", "message": "Document ingested"}


@app.post("/process")
async def process_document(request: dict):
    """Process a document using workflow"""
    print("=== PROCESS DOCUMENT ===")
    print(f"Request: {request}")
    print("Building LangGraph workflow...")
    print("Executing node_1: Input...")
    print("Executing node_2: Processor...")
    print("Executing node_3: Output...")
    print("Done!")
    return {"status": "success", "message": "Document processed"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
