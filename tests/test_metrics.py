# pyrefly: ignore [missing-import]
import pytest
from src.metrics import calculate_recall_at_k, calculate_mrr, calculate_precision_at_k

def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    ground_truth = ["doc2"]
    
    assert calculate_recall_at_k(retrieved, ground_truth, k=1) == 0.0
    assert calculate_recall_at_k(retrieved, ground_truth, k=2) == 1.0
    assert calculate_recall_at_k(retrieved, ground_truth, k=5) == 1.0

def test_mrr():
    retrieved = ["doc1", "doc2", "doc3"]
    ground_truth = ["doc2"]
    
    assert calculate_mrr(retrieved, ground_truth) == 0.5 # Rank 2 -> 1/2
    
    assert calculate_mrr(["doc2", "doc1"], ground_truth) == 1.0 # Rank 1
    assert calculate_mrr(["doc4", "doc5"], ground_truth) == 0.0 # Not found

def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    
    assert calculate_precision_at_k(retrieved, ["doc1", "doc2"], k=2) == 1.0
    assert calculate_precision_at_k(retrieved, ["doc2"], k=2) == 0.5
    assert calculate_precision_at_k(retrieved, ["doc4"], k=3) == 0.0
