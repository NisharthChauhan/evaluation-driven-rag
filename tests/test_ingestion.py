import os
import tempfile
# pyrefly: ignore [missing-import]
import pytest
from src.ingestion import DocumentIngestionPipeline, Document

@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a sample markdown file
        sample_md = os.path.join(tmpdirname, "sample.md")
        with open(sample_md, "w") as f:
            f.write("# Sample Document\n\nThis is a test.\n\n<!-- hidden comment -->\n\n\n\nToo many newlines.")
            
        # Create a subdirectory with another markdown file
        sub_dir = os.path.join(tmpdirname, "sub")
        os.makedirs(sub_dir)
        sub_md = os.path.join(sub_dir, "sub_sample.md")
        with open(sub_md, "w") as f:
            f.write("## Sub Document\n\nAnother test.")
            
        yield tmpdirname

def test_load_documents(temp_data_dir):
    pipeline = DocumentIngestionPipeline(temp_data_dir)
    documents = pipeline.load_documents()
    
    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    
    # Check cleaning
    sample_doc = next(doc for doc in documents if doc.source == "sample.md")
    assert "<!-- hidden comment -->" not in sample_doc.content
    assert "\n\n\n" not in sample_doc.content
    
    sub_doc = next(doc for doc in documents if "sub_sample.md" in doc.source)
    assert sub_doc.content == "## Sub Document\n\nAnother test."
