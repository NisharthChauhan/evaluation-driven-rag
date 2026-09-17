# Evaluation-Driven RAG System

An evaluation-driven Retrieval-Augmented Generation (RAG) framework for comparing vector, semantic, hybrid, and reranked retrieval pipelines using retrieval, answer-quality, and latency metrics.

## Overview

This project evaluates different RAG retrieval strategies on the **RAG-Multi-Corpus Benchmark**, a challenging and heterogeneous dataset containing 1,180 files across 5 synthetic enterprise domains, including ZX Bank, Velvera Technologies, and Aventro Motors.

The system compares four retrieval configurations:

- Simple Vector Retrieval
- Semantic Vector Retrieval
- Hybrid Retrieval
- Hybrid Retrieval + Cross-Encoder Reranking

The goal is to systematically measure retrieval quality, generated-answer quality, and latency rather than evaluating a RAG system only through qualitative examples.

## Key Features

- Multiple document formats: PDF, DOCX, PPTX, HTML, and Markdown
- Structure-aware semantic chunking
- Dense vector retrieval using ChromaDB
- Sparse keyword retrieval using BM25
- Hybrid retrieval with score normalization and weighted fusion
- Cross-encoder reranking
- Local LLM generation using Ollama
- Retrieval evaluation using:
  - Recall@5
  - Precision@5
  - Mean Reciprocal Rank (MRR)
- Answer evaluation using:
  - Faithfulness
  - Correctness
- Latency comparison across retrieval pipelines

## Problem Statement

> **How can we systematically improve and measure RAG retrieval performance on highly specific enterprise data?**

Naive RAG systems can suffer from missing context, poor chunk boundaries, and hallucination. General-purpose embedding models can also struggle with enterprise-specific terminology, acronyms, and exact identifiers.

This project addresses these challenges through reproducible retrieval metrics such as Recall@K and MRR, hybrid retrieval using vector search and BM25, score normalization and fusion, and LLM-as-a-judge evaluation.

## Architecture

```text
                    Documents
                       │
                       ▼
                Ingestion & Parsing
                       │
                       ▼
                 Chunking Strategy
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     ChromaDB                    BM25 Index
    Vector Search             Keyword Search
          │                         │
          │        ┌────────────────┘
          │        │
          ▼        ▼
       Vector     Hybrid
       Search     Fusion
          │        │
          └────┬───┘
               ▼
          Top-K Candidates
               │
               ▼
       Cross-Encoder Reranker
            (Optional)
               │
               ▼
            Qwen3:8b
               │
               ▼
          Final Answer
               │
               ▼
           Evaluation
```

## Technology Stack

- **Language**: Python 3.11
- **Vector Database**: ChromaDB (Local SQLite)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Keyword Search**: `rank_bm25` (BM25)
- **Reranker**: `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **LLM Engine**: Ollama running `qwen3:8b` locally
- **Metrics**: Custom Python implementations of Recall@5, MRR, Precision, and LLM-as-a-judge

## Project Structure

```text
evaluation-driven-rag/
│
├── .github/
│   └── workflows/
│       └── evaluate.yml
│
├── data/
│   └── benchmark.json
│
├── results/
│   └── metrics.csv
│
├── scripts/
│   ├── generate_benchmark.py
│   └── run_experiments.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── bm25_store.py
│   ├── chunking.py
│   ├── generator.py
│   ├── hybrid_store.py
│   ├── ingestion.py
│   ├── metrics.py
│   └── vector_store.py
│
├── tests/
│   ├── __init__.py
│   ├── test_chunking.py
│   ├── test_generation.py
│   ├── test_hybrid.py
│   ├── test_ingestion.py
│   ├── test_metrics.py
│   └── test_retrieval.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NisharthChauhan/evaluation-driven-rag.git
cd evaluation-driven-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Linux/macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## LLM Setup

This project uses a local LLM through **Ollama** for generation and LLM-based evaluation.

Install Ollama and download the required model:

```bash
ollama pull qwen3:8b
```

Ensure that Ollama is running before executing the RAG pipeline.

## Usage

### 1. Run the benchmark experiments

```bash
python scripts/run_experiments.py
```

> If the evaluation configuration requires an external API key, configure it using an environment variable. **Do not place API keys directly in the source code or README.**

### 2. Start the RAG API

```bash
uvicorn src.api:app --reload
```

The API can then be accessed through the local server address shown by Uvicorn.

> The exact experiment and API commands should be kept consistent with the current project implementation and configuration.

## Evaluation Methodology

The project evaluates four distinct retrieval configurations on a curated benchmark of enterprise questions:

1. **Baseline**: Fixed-size chunking (500 chars) + Vector Search (Chroma).
2. **Semantic Vector**: Markdown-aware semantic chunking + Vector Search.
3. **Hybrid RAG**: Semantic chunking (size = 1000) + Vector Search + BM25 score fusion.
4. **Advanced RAG**: Semantic chunking + Hybrid Retrieval + Cross-Encoder Reranking.

## Results

The reported results are based on a curated set of 50 enterprise questions selected from the RAG-Multi-Corpus Benchmark.

| Method | Recall@5 | Precision@5 | MRR | Faithfulness | Correctness | Latency (s) |
|--------|---------:|------------:|----:|-------------:|------------:|------------:|
| Baseline (Simple Vector) | 0.42 | 0.38 | 0.36 | 0.65 | 0.40 | 1.01 |
| Semantic Vector (1000) | 0.68 | 0.55 | 0.58 | 0.82 | 0.62 | 1.02 |
| **Hybrid (1000, α = 0.5)** | **0.88** | **0.78** | **0.74** | **0.94** | **0.86** | **1.03** |
| Advanced (Reranker) | 0.76 | 0.65 | 0.62 | 0.89 | 0.74 | 1.82 |

## Key Findings

1. **Semantic chunking with 1000-character blocks substantially improved retrieval and answer-quality metrics compared with the 500-character baseline in this benchmark.**

2. **Hybrid search substantially improved retrieval performance over pure vector search on the evaluated enterprise benchmark.** Combining semantic vector retrieval with BM25 provided better handling of enterprise-specific terminology and exact identifiers such as `"ZX Bank Form 104-B"`.

3. **The generic MS-MARCO-trained reranker reduced retrieval performance on this benchmark, suggesting a domain mismatch between its training data and the enterprise-specific queries and documents.** The reranked configuration also increased latency compared with the hybrid configuration.

## Limitations

- The current evaluation benchmark is limited to 50 questions and may not capture all possible edge cases.
- Enterprise-specific acronyms and terminology can remain challenging for general-purpose embedding models, particularly when relevant matches depend on exact terminology.
- The benchmark consists of synthetic enterprise data, so the observed results may not directly generalize to real-world enterprise datasets.

## Future Work

- Domain-specific reranking through fine-tuning.
- Adaptive weighting for hybrid retrieval.
- Automated retrieval regression testing through CI/CD.

## Dataset / Benchmark Attribution

This project uses the **RAG-Multi-Corpus Dataset** for evaluating retrieval and RAG answer quality.

- **Original Source:** [RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus)
- The dataset is synthetic and contains fictional enterprise organizations.
- The dataset is an external resource and is not developed as part of this project.
- The original benchmark contains a larger set of queries; this project uses a curated subset of 50 questions for its evaluation experiments.
- To reproduce the experiments, download the dataset from the original repository and place the required files in the appropriate `data/` directory.

## Author

**Nisharth Chauhan**
