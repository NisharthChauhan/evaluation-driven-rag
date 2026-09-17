import json
import os
import random
from typing import List, Dict, Any

from src.ingestion import DocumentIngestionPipeline
from src.chunking import SemanticChunker

def generate_benchmark_dataset(data_dir: str, output_path: str, num_questions: int = 50):
    """
    In a real scenario, this script would iterate over extracted sections,
    send them to an LLM (e.g., GPT-4) and ask it to generate specific types of questions
    (Factual, Conceptual, Technical, Exact-term).
    
    Here, we provide a placeholder generation to demonstrate reproducibility.
    """
    pipeline = DocumentIngestionPipeline(data_dir)
    documents = pipeline.load_documents()
    
    chunker = SemanticChunker(max_chunk_size=500)
    chunks = chunker.chunk_documents(documents)
    
    if not chunks:
        print("No chunks found in data directory. Benchmark will be empty.")
        return
        
    benchmark = []
    for i in range(num_questions):
        # Pick a random chunk for context
        target_chunk = random.choice(chunks)
        
        # Determine question type
        q_type = random.choice(["Factual", "Conceptual", "Technical", "Exact-term", "Multi-hop"])
        
        # Provide a placeholder structure
        # In reality: prompt LLM to generate question & ground truth from target_chunk
        item = {
            "id": f"q{str(i+1).zfill(3)}",
            "question": f"Sample {q_type} question for {target_chunk.metadata.get('source')}?",
            "ground_truth_answer": f"This is the ground truth based on {target_chunk.metadata.get('section', 'unknown')}.",
            "source_documents": [target_chunk.metadata.get("source")],
            "source_sections": [target_chunk.metadata.get("section")],
            "expected_chunk_id": target_chunk.chunk_id, # useful for Retrieval evaluation
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "category": q_type
        }
        benchmark.append(item)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=4)
        
    print(f"Successfully generated {num_questions} questions at {output_path}")

if __name__ == "__main__":
    generate_benchmark_dataset(
        data_dir="./data/fastapi_docs",
        output_path="./data/benchmark.json",
        num_questions=50
    )
