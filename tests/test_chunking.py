# pyrefly: ignore [missing-import]
import pytest
from src.ingestion import Document
from src.chunking import SimpleChunker, SemanticChunker

@pytest.fixture
def sample_documents():
    return [
        Document(
            document_id="doc1",
            source="doc1.md",
            content="This is a simple document. It has some text. " * 20, # length: 45 * 20 = 900
            metadata={"document_id": "doc1", "source": "doc1.md"}
        ),
        Document(
            document_id="doc2",
            source="doc2.md",
            content="# Header 1\n\nParagraph 1.\n\n## Header 2\n\nParagraph 2.\n\nParagraph 3.",
            metadata={"document_id": "doc2", "source": "doc2.md"}
        )
    ]

def test_simple_chunker(sample_documents):
    chunker = SimpleChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_documents([sample_documents[0]])
    
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert chunks[0].document_id == "doc1"
    
def test_semantic_chunker(sample_documents):
    chunker = SemanticChunker(max_chunk_size=100)
    chunks = chunker.chunk_documents([sample_documents[1]])
    
    assert len(chunks) > 0
    # Check if section metadata is preserved
    header1_chunk = next(chunk for chunk in chunks if chunk.metadata.get("section") == "Header 1")
    assert "Paragraph 1" in header1_chunk.content
    
    header2_chunk = next(chunk for chunk in chunks if chunk.metadata.get("section") == "Header 2")
    assert "Paragraph 2" in header2_chunk.content
