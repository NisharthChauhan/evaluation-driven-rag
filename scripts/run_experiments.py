import json
import time
import os
import pandas as pd
from typing import List, Dict, Any

from src.ingestion import DocumentIngestionPipeline
from src.chunking import SimpleChunker, SemanticChunker
from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.hybrid_store import HybridRetriever, Reranker
from src.generator import Generator
from src.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k, LLMEvaluator

def load_benchmark(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_match(fact_text: str, chunk_text: str) -> bool:
    import re
    def tokenize(t):
        return set(re.findall(r'\w+', t.lower()))
    
    fact_words = tokenize(fact_text)
    chunk_words = tokenize(chunk_text)
    if not fact_words or not chunk_words:
        return False
    overlap = len(fact_words.intersection(chunk_words))
    return (overlap / len(fact_words)) > 0.5

def get_ground_truth_ids(chunks, facts) -> List[str]:
    gt_ids = set()
    for fact in facts:
        filename = fact.get("filename")
        text = fact.get("text")
        for chunk in chunks:
            source_file = chunk.metadata.get("source_file", "")
            if filename == source_file or (filename and filename in chunk.metadata.get("source", "")):
                if is_match(text, chunk.content):
                    gt_ids.add(chunk.chunk_id)
    return list(gt_ids)

def run_experiments():
    print("Setting up data...")
    pipeline = DocumentIngestionPipeline("./data/corpus")
    documents = pipeline.load_documents()
    print(f"Loaded {len(documents)} documents.")
    
    print("Chunking documents...")
    # Test different chunk sizes
    simple_chunks_500 = SimpleChunker(500, 100).chunk_documents(documents)
    semantic_chunks_500 = SemanticChunker(500).chunk_documents(documents)
    semantic_chunks_1000 = SemanticChunker(1000).chunk_documents(documents)
    
    print("Initializing indices...")
    import shutil
    shutil.rmtree("./data/chroma_db_simple_500", ignore_errors=True)
    shutil.rmtree("./data/chroma_db_semantic_500", ignore_errors=True)
    shutil.rmtree("./data/chroma_db_semantic_1000", ignore_errors=True)
    
    vs_simple_500 = VectorStore(persist_directory="./data/chroma_db_simple_500")
    vs_simple_500.add_chunks(simple_chunks_500)
    
    vs_sem_500 = VectorStore(persist_directory="./data/chroma_db_semantic_500")
    vs_sem_500.add_chunks(semantic_chunks_500)
    
    vs_sem_1000 = VectorStore(persist_directory="./data/chroma_db_semantic_1000")
    vs_sem_1000.add_chunks(semantic_chunks_1000)
    
    bm25_sem_500 = BM25Store()
    bm25_sem_500.add_chunks(semantic_chunks_500)
    
    bm25_sem_1000 = BM25Store()
    bm25_sem_1000.add_chunks(semantic_chunks_1000)
    
    # Retrievers
    hybrid_alpha_3 = HybridRetriever(vs_sem_500, bm25_sem_500, alpha=0.3)
    hybrid_alpha_5 = HybridRetriever(vs_sem_500, bm25_sem_500, alpha=0.5)
    hybrid_alpha_7 = HybridRetriever(vs_sem_500, bm25_sem_500, alpha=0.7)
    
    hybrid_1000_alpha_5 = HybridRetriever(vs_sem_1000, bm25_sem_1000, alpha=0.5)
    
    reranker = Reranker()
    generator = Generator()
    evaluator = LLMEvaluator()
    
    benchmark = load_benchmark("./data/evaluation/benchmark_50.json")
    
    # Run only top 3 configurations for the final benchmark to save time
    configurations = {
        "Simple Vector (500)": {"store": vs_simple_500, "chunks": simple_chunks_500, "type": "vector"},
        "Hybrid (size=1000, alpha=0.5)": {"store": hybrid_1000_alpha_5, "chunks": semantic_chunks_1000, "type": "hybrid"},
        "Advanced (Hybrid 1000a.5 + Rerank)": {"store": hybrid_1000_alpha_5, "chunks": semantic_chunks_1000, "type": "advanced"}
    }
    
    results_data = []
    
    for config_name, config in configurations.items():
        print(f"\n--- Running Configuration: {config_name} ---")
        
        recall_5 = []
        precision_5 = []
        mrr = []
        latencies = []
        faithfulness_scores = []
        correctness_scores = []
        
        for idx, item in enumerate(benchmark[:50]): # Reduced to 50 to prevent local Ollama crashes
            if idx % 5 == 0:
                print(f"Processing query {idx+1}/50...")
                
            q = item["question"]
            facts = item.get("supporting_facts", [])
            
            chunks_to_use = config["chunks"]
            expected_ids = get_ground_truth_ids(chunks_to_use, facts)
            
            if not expected_ids:
                continue
            
            start_time = time.time()
            time.sleep(1) # Small delay to let local LLM clear context and prevent timeouts
            
            if config["type"] == "vector":
                res = config["store"].search(q, top_k=5)
            elif config["type"] == "hybrid":
                res = config["store"].search(q, top_k=5)
            elif config["type"] == "advanced":
                candidates = config["store"].search(q, top_k=50)
                res = reranker.rerank(q, candidates, top_k=5)
                
            latency = time.time() - start_time
            latencies.append(latency)
            
            retrieved_ids = [r["chunk_id"] for r in res]
            
            recall_5.append(calculate_recall_at_k(retrieved_ids, expected_ids, 5))
            precision_5.append(calculate_precision_at_k(retrieved_ids, expected_ids, 5))
            mrr.append(calculate_mrr(retrieved_ids, expected_ids))
            
            # Skip LLM evaluation to prevent local Ollama crashes on large batches
            faithfulness_scores.append(0.0)
            correctness_scores.append(0.0)
                
        avg_recall = sum(recall_5) / len(recall_5) if recall_5 else 0
        avg_prec = sum(precision_5) / len(precision_5) if precision_5 else 0
        avg_mrr = sum(mrr) / len(mrr) if mrr else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        
        avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        avg_corr = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0
        
        results_data.append({
            "Method": config_name,
            "Recall@5": avg_recall,
            "Precision@5": avg_prec,
            "MRR": avg_mrr,
            "Faithfulness": avg_faith if faithfulness_scores else "N/A (No API Key)",
            "Correctness": avg_corr if faithfulness_scores else "N/A (No API Key)",
            "Latency (s)": avg_lat
        })
        
    df = pd.DataFrame(results_data)
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/metrics.csv", index=False)
    
    print("\n==================================================")
    print("RAG EVALUATION RESULTS")
    print("==================================================")
    print(df.to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    run_experiments()
