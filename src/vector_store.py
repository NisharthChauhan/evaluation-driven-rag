from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from chromadb.utils import embedding_functions
from src.chunking import Chunk

class VectorStore:
    def __init__(self, collection_name: str = "rag_collection", model_name: str = "BAAI/bge-small-en-v1.5", persist_directory: str = "./data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        import os
        base_url = os.environ.get("OLLAMA_BASE_URL")
        if base_url:
            # Convert http://localhost:11434/v1 to http://localhost:11434/api/embeddings
            ollama_url = base_url.replace("/v1", "") + "/api/embeddings"
            self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
                url=ollama_url,
                model_name="nomic-embed-text"
            )
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function # type: ignore
        )
        
    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
            
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Chroma handles batching, but we can batch it just in case
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size] # type: ignore
            )
            
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                content = results['documents'][0][i] if results.get('documents') and results['documents'] else ""
                metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'] else {}
                distance = results['distances'][0][i] if results.get('distances') and results['distances'] else 0.0
                
                formatted_results.append({
                    "chunk_id": results['ids'][0][i],
                    "content": content,
                    "metadata": metadata,
                    "distance": distance
                })
                
        return formatted_results
