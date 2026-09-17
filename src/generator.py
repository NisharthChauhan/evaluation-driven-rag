from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from openai import OpenAI
import os

class Generator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Relies on OPENAI_API_KEY environment variable
        self.model_name = model_name
        self.client = None
        
        base_url = os.environ.get("OLLAMA_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if base_url:
            # Connect to local Ollama server using OpenAI compatibility layer
            self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=300.0)
            # If using ollama, use the user's downloaded qwen3 model
            if model_name == "gpt-4o-mini":
                self.model_name = "qwen3:8b"
        elif api_key:
            self.client = OpenAI(api_key=api_key, timeout=300.0)
        
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.client:
            return {"answer": "No OpenAI API key found. Generation skipped.", "sources": []}
        context_text = "\n\n".join([
            f"[Source: {chunk['metadata'].get('source', 'Unknown')}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""You are a technical assistant. Answer the user's question based strictly on the provided context.
        
Context:
{context_text}

Question:
{query}

Instructions:
1. Use only the information provided in the context.
2. If the context does not contain enough information to answer the question, say "The provided documentation does not contain enough information to answer this question."
3. Cite your sources using the format [Source: source_name].
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        answer = response.choices[0].message.content
        sources = [chunk["metadata"].get("source", "Unknown") for chunk in context_chunks]
        
        return {
            "answer": answer,
            "sources": list(set(sources)) # Deduplicate sources
        }
