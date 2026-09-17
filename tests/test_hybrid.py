# pyrefly: ignore [missing-import]
import pytest
import os
import shutil
from src.chunking import Chunk
from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.hybrid_store import HybridRetriever, Reranker

@pytest.fixture
def sample_chunks():
    return [
        Chunk("c1", "doc1", "FastAPI is a fast web framework.", {"source": "d1"}),
        Chunk("c2", "doc2", "Kubernetes automates deployment.", {"source": "d2"}),
        Chunk("c3", "doc3", "Pydantic is for data validation.", {"source": "d3"}),
        Chunk("c4", "doc4", "OAuth2 is an authentication protocol often used with FastAPI.", {"source": "d4"})
    ]

@pytest.fixture
def stores(sample_chunks, tmp_path):
    db_path = str(tmp_path / "chroma_db_hybrid")
    vec_store = VectorStore(persist_directory=db_path)
    bm25_store = BM25Store()
    
    vec_store.add_chunks(sample_chunks)
    bm25_store.add_chunks(sample_chunks)
    
    yield vec_store, bm25_store
    
    shutil.rmtree(db_path, ignore_errors=True)

def test_hybrid_search(stores):
    vec_store, bm25_store = stores
    retriever = HybridRetriever(vec_store, bm25_store, alpha=0.5)
    
    results = retriever.search("FastAPI authentication", top_k=2)
    assert len(results) == 2
    # The combined score logic should run without errors
    assert "hybrid_score" in results[0]
    
def test_reranker(stores):
    vec_store, bm25_store = stores
    retriever = HybridRetriever(vec_store, bm25_store, alpha=0.5)
    results = retriever.search("FastAPI authentication", top_k=4)
    
    reranker = Reranker()
    reranked = reranker.rerank("FastAPI authentication", results, top_k=2)
    
    assert len(reranked) == 2
    assert "reranker_score" in reranked[0]
