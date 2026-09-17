from typing import List, Dict, Any

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    if not ground_truth_ids:
        return 0.0
    hits = set(retrieved_k).intersection(set(ground_truth_ids))
    # True recall: proportion of ground truth documents retrieved
    return len(hits) / float(len(ground_truth_ids))

def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    # MRR is 1 / (rank of first relevant document)
    for rank, ret_id in enumerate(retrieved_ids, 1):
        if ret_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0

def calculate_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]

    if not retrieved_k:
        return 0.0

    hits = set(retrieved_k).intersection(set(ground_truth_ids))

    # Precision@K = relevant retrieved documents / retrieved documents
    return len(hits) / float(len(retrieved_k))

class LLMEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        import os
        # pyrefly: ignore [missing-import]
        from openai import OpenAI
        self.model_name = model_name
        self.client = None
        
        base_url = os.environ.get("OLLAMA_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if base_url:
            self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=300.0)
            if model_name == "gpt-4o-mini":
                self.model_name = "qwen3:8b"
        elif api_key:
            self.client = OpenAI(api_key=api_key, timeout=300.0)

    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        if not self.client:
            return {"faithful": False, "reason": "No API key"}
        prompt = f"""You are an evaluator. Determine if the given Answer is faithful to the Context.
An answer is faithful if it does not contain any hallucinations or information outside the context.
Answer with a JSON object: {{"faithful": bool, "reason": "str"}}

Context: {context}
Question: {question}
Answer: {answer}
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        import json
        try:
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            return {"faithful": False, "reason": "Failed to parse evaluation"}
            
    def evaluate_correctness(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        if not self.client:
            return {"correct": False, "reason": "No API key"}
        prompt = f"""You are an evaluator. Determine if the given Answer correctly answers the Question based on the Ground Truth.
Answer with a JSON object: {{"correct": bool, "reason": "str"}}

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        import json
        try:
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            return {"correct": False, "reason": "Failed to parse evaluation"}
