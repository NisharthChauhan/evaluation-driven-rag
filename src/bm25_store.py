from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
import numpy as np
from src.chunking import Chunk

class BM25Store:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: BM25Okapi = None
        self.corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace and punctuation tokenization
        # In a production system, this could use NLTK or SpaCy
        import string
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
            
        self.chunks.extend(chunks)
        new_tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        self.corpus.extend(new_tokenized_corpus)
        
        # Re-initialize BM25 with the full corpus
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only return if there is some match
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "score": scores[idx]
                })
                
        return results
