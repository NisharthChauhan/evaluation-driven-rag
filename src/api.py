# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import List, Optional
import os

from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.hybrid_store import HybridRetriever, Reranker
from src.generator import Generator

app = FastAPI(title="Evaluation-Driven RAG API")

# Initialize global stores/models 
# Note: In a real prod setup, these would be initialized via lifespan/startup events
# and injected as dependencies, but this keeps the script simple for demo.
# Ensure you run the ingestion/indexing script before querying this API!
from src.ingestion import DocumentIngestionPipeline
from src.chunking import SemanticChunker

try:
    print("Loading Vector Store...")
    vec_store = VectorStore(persist_directory="./data/chroma_db_semantic")
    
    print("Building BM25 Index...")
    bm25_store = BM25Store()
    pipeline = DocumentIngestionPipeline("./data/fastapi_docs")
    documents = pipeline.load_documents()
    chunks = SemanticChunker(max_chunk_size=500).chunk_documents(documents)
    bm25_store.add_chunks(chunks)
    
    hybrid_retriever = HybridRetriever(vec_store, bm25_store)
    reranker = Reranker()
    generator = Generator()
    print("All models initialized successfully!")
except Exception as e:
    print(f"Warning: Failed to initialize models. Make sure data is ingested. Error: {e}")

class QueryRequest(BaseModel):
    question: str
    retrieval_method: str = "hybrid_rerank" # Options: vector, bm25, hybrid, hybrid_rerank

class SourceInfo(BaseModel):
    document: str
    section: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    retrieval_method: str

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    method = request.retrieval_method
    top_k = 5
    
    if method == "vector":
        retrieved_chunks = vec_store.search(request.question, top_k=top_k)
    elif method == "bm25":
        retrieved_chunks = bm25_store.search(request.question, top_k=top_k)
    elif method == "hybrid":
        retrieved_chunks = hybrid_retriever.search(request.question, top_k=top_k)
    elif method == "hybrid_rerank":
        retrieved_chunks = hybrid_retriever.search(request.question, top_k=20)
        retrieved_chunks = reranker.rerank(request.question, retrieved_chunks, top_k=top_k)
    else:
        raise HTTPException(status_code=400, detail="Invalid retrieval method")
        
    generation_result = generator.generate_answer(request.question, retrieved_chunks)
    
    # Format sources for response
    sources = []
    for chunk in retrieved_chunks:
        source_info = SourceInfo(
            document=chunk["metadata"].get("source", "Unknown"),
            section=chunk["metadata"].get("section")
        )
        # Avoid duplicate SourceInfos
        if source_info not in sources:
            sources.append(source_info)
    
    return QueryResponse(
        answer=generation_result["answer"],
        sources=sources,
        retrieval_method=method
    )
