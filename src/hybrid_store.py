from typing import List, Dict, Any
import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, vector_store, bm25_store, alpha: float = 0.5):
        """
        alpha: weight for vector score. (1-alpha) is for BM25 score.
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.alpha = alpha
        
    def _normalize_scores(self, results: List[Dict[str, Any]], score_key: str, inverse: bool = False) -> List[Dict[str, Any]]:
        if not results:
            return results
            
        scores = [res[score_key] for res in results]
        
        # Chroma returns L2 distances by default, where lower is better. 
        # If inverse is True, we invert it so higher is better.
        if inverse:
            # simple inversion: max_distance - distance
            max_score = max(scores)
            scores = [max_score - s for s in scores]
            
        min_score = min(scores)
        max_score = max(scores)
        
        # Min-max normalization (0 to 1)
        if max_score - min_score == 0:
            normalized = [1.0] * len(scores)
        else:
            normalized = [(s - min_score) / (max_score - min_score) for s in scores]
            
        for i, res in enumerate(results):
            res[f"normalized_{score_key}"] = normalized[i]
            
        return results

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Get more candidates initially
        candidate_k = max(top_k * 2, 20)
        
        vec_results = self.vector_store.search(query, top_k=candidate_k)
        bm25_results = self.bm25_store.search(query, top_k=candidate_k)
        
        # Normalize scores (invert L2 distances from Chroma so higher is better)
        vec_results = self._normalize_scores(vec_results, "distance", inverse=True)
        bm25_results = self._normalize_scores(bm25_results, "score", inverse=False)
        
        # Create a unified map of chunk_id -> combined score
        combined_scores = {}
        chunk_data = {}
        
        for res in vec_results:
            cid = res["chunk_id"]
            combined_scores[cid] = self.alpha * res.get("normalized_distance", 0.0)
            chunk_data[cid] = res
            
        for res in bm25_results:
            cid = res["chunk_id"]
            if cid in combined_scores:
                combined_scores[cid] += (1 - self.alpha) * res.get("normalized_score", 0.0)
            else:
                combined_scores[cid] = (1 - self.alpha) * res.get("normalized_score", 0.0)
                chunk_data[cid] = res
                
        # Sort by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for cid, score in sorted_chunks[:top_k]:
            res = chunk_data[cid].copy()
            res["hybrid_score"] = score
            final_results.append(res)
            
        return final_results


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not results:
            return []
            
        pairs = [[query, res["content"]] for res in results]
        scores = self.model.predict(pairs)
        
        for i, res in enumerate(results):
            res["reranker_score"] = float(scores[i])
            
        # Sort by reranker score
        reranked_results = sorted(results, key=lambda x: x["reranker_score"], reverse=True)
        return reranked_results[:top_k]
