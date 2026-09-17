from typing import List, Dict, Any
from dataclasses import dataclass
import re
from src.ingestion import Document

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]

class SimpleChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            text = doc.content
            start = 0
            chunk_index = 0
            
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                chunk_id = f"{doc.document_id}_chunk_{chunk_index}"
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    content=chunk_text,
                    metadata=chunk_metadata
                ))
                
                start += (self.chunk_size - self.chunk_overlap)
                chunk_index += 1
                
        return chunks

class SemanticChunker:
    """
    Structure-aware chunker for Markdown files.
    Splits mainly on headers and double newlines (paragraphs/blocks).
    Groups small blocks together up to a soft maximum size.
    """
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            # Split by headers (e.g. \n## Header)
            sections = re.split(r'(?=\n#{1,6} )', doc.content)
            
            chunk_index = 0
            current_section_title = ""
            
            for section in sections:
                if not section.strip():
                    continue
                
                # Extract header if present
                header_match = re.match(r'^\n?(#{1,6})\s+(.*)', section)
                if header_match:
                    current_section_title = header_match.group(2).strip()
                
                # Split section into smaller blocks if it's too large
                blocks = re.split(r'\n\n+', section)
                current_chunk_text = ""
                
                for block in blocks:
                    block = block.strip()
                    if not block:
                        continue
                        
                    if len(current_chunk_text) + len(block) + 2 <= self.max_chunk_size:
                        current_chunk_text += block + "\n\n"
                    else:
                        if current_chunk_text.strip():
                            chunks.append(self._create_chunk(doc, chunk_index, current_chunk_text.strip(), current_section_title))
                            chunk_index += 1
                        current_chunk_text = block + "\n\n"
                        
                        # If a single block is still larger than max, split it roughly by chunk_size
                        if len(current_chunk_text) > self.max_chunk_size:
                            large_text = current_chunk_text.strip()
                            start = 0
                            overlap = int(self.max_chunk_size * 0.2)
                            while start < len(large_text):
                                end = start + self.max_chunk_size
                                sub_chunk = large_text[start:end]
                                chunks.append(self._create_chunk(doc, chunk_index, sub_chunk, current_section_title))
                                chunk_index += 1
                                start += (self.max_chunk_size - overlap)
                            current_chunk_text = ""
                            
                if current_chunk_text.strip():
                    chunks.append(self._create_chunk(doc, chunk_index, current_chunk_text.strip(), current_section_title))
                    chunk_index += 1
                    
        return chunks

    def _create_chunk(self, doc: Document, index: int, text: str, section_title: str) -> Chunk:
        chunk_id = f"{doc.document_id}_chunk_{index}"
        chunk_metadata = doc.metadata.copy()
        chunk_metadata["chunk_index"] = index
        if section_title:
            chunk_metadata["section"] = section_title
            # Prepend section title to content for better context in embedding
            text = f"Section: {section_title}\n\n{text}"
            
        return Chunk(
            chunk_id=chunk_id,
            document_id=doc.document_id,
            content=text,
            metadata=chunk_metadata
        )
