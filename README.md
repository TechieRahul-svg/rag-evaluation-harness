🚀 RAG Evaluation Harness  
### An Evaluation-Driven Retrieval-Augmented Generation System with Hybrid Search, Re-ranking & Automated Quality Assessment

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-blue)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![RAG](https://img.shields.io/badge/AI-RAG-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

**RAG Evaluation Harness** is a production-oriented Retrieval-Augmented Generation (RAG) system designed to evaluate, benchmark, and improve retrieval quality in AI applications.

Unlike traditional RAG chatbot implementations that only focus on generating answers, this project focuses on the most critical part of RAG systems:

> **"Can the system retrieve the correct information before generating an answer?"**

The system benchmarks multiple retrieval strategies, evaluates their performance using ground-truth datasets, applies advanced ranking techniques, and measures final answer quality using automated evaluation metrics.

---

# 🎯 Problem Statement

Large organizations maintain thousands of documents containing:

- Technical documentation
- HR policies
- Legal documents
- Financial reports
- Internal knowledge bases

Passing all documents directly to an LLM is:

- Expensive
- Slow
- Limited by context windows

A RAG system solves this problem by:

```
User Query
     |
     ↓
Retrieve Relevant Documents
     |
     ↓
Provide Context to LLM
     |
     ↓
Generate Grounded Answer
```

However, traditional RAG systems often fail because:

- Incorrect documents are retrieved
- Similar topics confuse embeddings
- Chunking strategy affects retrieval quality
- LLM generates confident hallucinations

This project solves these problems through **retrieval evaluation and optimization**.

---

# 🏗️ System Architecture


```
                    Documents
                        |
                        ↓
              Document Processing
                        |
                        ↓
          ┌─────────────┴─────────────┐
          ↓                           ↓
   Chunking Strategies          Metadata Processing
          |
          |
 ┌────────┼─────────┐
 ↓        ↓         ↓
Fixed  Recursive  Semantic
Chunk   Chunk     Chunk
 |
 ↓
Embedding Generation
 |
 ↓
ChromaDB Vector Store


                User Query
                    |
                    ↓

        ┌─────────────────────┐
        │ Hybrid Retrieval     │
        │ BM25 + Vector Search │
        └─────────────────────┘

                    |
                    ↓

        Reciprocal Rank Fusion

                    |
                    ↓

          Cross Encoder Reranker

                    |
                    ↓

              Relevant Context

                    |
                    ↓

          LLM Generation Layer

          Groq Llama 3.3
              |
              ↓

          Final Answer

                    |
                    ↓

          RAGAS Evaluation
```

---

# ✨ Key Features

## 1. Multi-Strategy Document Chunking

Implemented and compared:

### Fixed Chunking

Splits documents into fixed-size chunks.

### Recursive Chunking

Preserves document structure using hierarchical splitting.

### Semantic Chunking

Uses embedding similarity to create meaning-based chunks.

Improved semantic chunking by:

- Preserving Markdown headings
- Maintaining code blocks
- Improving contextual boundaries


---

# 2. Retrieval Benchmarking

Created a custom evaluation dataset containing:

- 20 benchmark questions
- Verified answers
- Source document references

Metrics:

| Metric | Description |
|-|-|
| Hit@1 | Correct document retrieved at rank 1 |
| Hit@3 | Correct document retrieved within top 3 |
| Hit@5 | Correct document retrieved within top 5 |


Example:

```
Question
    |
Retriever
    |
Top-K Documents
    |
Compare with Ground Truth
    |
Calculate Retrieval Accuracy
```

---

# 3. Hybrid Retrieval System

Implemented hybrid search combining:

## Dense Retrieval

Using:

- Sentence Transformers
- ChromaDB Vector Search


## Sparse Retrieval

Using:

- BM25 keyword retrieval


Combined using:

## Reciprocal Rank Fusion (RRF)

Benefits:

- Captures semantic similarity
- Preserves exact keyword matching
- Improves retrieval robustness


---

# 4. Cross Encoder Re-ranking

Vector similarity alone may retrieve related but incorrect documents.

Implemented re-ranking pipeline:

```
Retrieve Top 20 Candidates

          ↓

Cross Encoder Model

          ↓

Rank by Relevance Score

          ↓

Select Top 5 Contexts
```

This improves:

- Precision
- Hit@1 accuracy
- Context relevance


---

# 5. End-to-End RAG Generation

Implemented complete RAG pipeline:

```
Question

 ↓

Hybrid Retriever

 ↓

Reranker

 ↓

Relevant Context

 ↓

LLM

 ↓

Answer
```

Supported Generators:

- Groq Llama 3.3
- Local Qwen fallback


---

# 6. Automated RAG Evaluation

Implemented LLM-based evaluation using RAGAS metrics:

Evaluates:

## Faithfulness

Does the answer remain grounded in retrieved context?

## Answer Relevance

Does the generated response answer the user's question?


---

# 📊 Benchmark Results

## Retrieval Performance (20 Questions)

| Strategy | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|
| Fixed Chunking | 65% | 100% | 100% |
| Recursive Chunking | **70%** | 80% | 100% |
| Semantic Chunking | 65% | 90% | 95% |


---

# 🛠️ Tech Stack

## Backend

- Python
- FastAPI
- LangChain
- ChromaDB

## Retrieval

- Sentence Transformers
- BM25
- Reciprocal Rank Fusion
- Cross Encoder Re-ranking

## LLM

- Groq Llama 3.3
- HuggingFace Transformers

## Evaluation

- Custom Retrieval Evaluation Harness
- RAGAS

## Frontend

- React.js


---

# 📂 Project Structure

```
rag-evaluation-harness/

│
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── retrieval/
│   ├── vectorstore/
│   ├── evaluation/
│
├── backend/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│
├── frontend/
│
├── evaluation/
│   ├── questions.json
│   ├── retrieval_results.json
│
├── tests/
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/<username>/rag-evaluation-harness.git

cd rag-evaluation-harness
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create:

```
.env
```

Add:

```env
GROQ_API_KEY=your_api_key
```

---

# ▶️ Running the Application


## Start Backend

```bash
uvicorn backend.main:app --reload
```


Backend:

```
http://127.0.0.1:8000
```


API Docs:

```
http://127.0.0.1:8000/docs
```


---

## Start Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# 🧪 Running Evaluation

Run retrieval evaluation:

```bash
python src/evaluation/run_retrieval_evaluation.py
```


Run RAG quality evaluation:

```bash
python src/evaluation/ragas_eval.py
```

---

# 🔬 Future Improvements

- [ ] Deploy using Docker + Kubernetes
- [ ] Add PostgreSQL/PGVector support
- [ ] Add multi-document upload support
- [ ] Add authentication
- [ ] Add CI/CD regression testing
- [ ] Expand benchmark dataset to 100+ questions


---

# 💡 Key Learnings

Through this project:

- Learned production RAG architecture
- Compared retrieval strategies experimentally
- Designed evaluation datasets
- Improved semantic chunking techniques
- Implemented hybrid search systems
- Applied re-ranking for retrieval optimization
- Evaluated LLM outputs using automated metrics


---

# 👨‍💻 Author

**Rahul Panda**

B.Tech Computer Science Engineering

Interested in:
- Artificial Intelligence
- Machine Learning
- Generative AI
- Retrieval-Augmented Generation


---
