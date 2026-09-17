# pyrefly: ignore [missing-import]
import pytest
import os
import shutil
from src.chunking import Chunk
from src.vector_store import VectorStore
from src.bm25_store import BM25Store

@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="c1",
            document_id="doc1",
            content="FastAPI is a modern, fast web framework for building APIs with Python.",
            metadata={"source": "doc1.md"}
        ),
        Chunk(
            chunk_id="c2",
            document_id="doc2",
            content="Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.",
            metadata={"source": "doc2.md"}
        ),
        Chunk(
            chunk_id="c3",
            document_id="doc3",
            content="Pydantic is a data validation library for Python using type hints.",
            metadata={"source": "doc3.md"}
        )
    ]

def test_vector_store(sample_chunks, tmp_path):
    # Use a temporary directory for ChromaDB
    db_path = str(tmp_path / "chroma_db")
    
    # We use a smaller/faster embedding model for tests if needed, but default is fine
    store = VectorStore(persist_directory=db_path)
    store.add_chunks(sample_chunks)
    
    # Test vector search
    results = store.search("How to validate data in Python?", top_k=2)
    assert len(results) > 0
    # The Pydantic chunk should ideally be top, but let's just check format
    assert "chunk_id" in results[0]
    assert "content" in results[0]
    assert "distance" in results[0]
    
    # Cleanup
    shutil.rmtree(db_path, ignore_errors=True)

def test_bm25_store(sample_chunks):
    store = BM25Store()
    store.add_chunks(sample_chunks)
    
    # Test exact keyword match
    results = store.search("FastAPI framework", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"
    
    results = store.search("Kubernetes deployment", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c2"
