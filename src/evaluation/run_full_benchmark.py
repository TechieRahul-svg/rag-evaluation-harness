"""
Unified RAG Benchmark Evaluation

Compares:
1. Fixed Chunking
2. Semantic Chunking
3. Hybrid Search
4. Hybrid + Re-ranking

Evaluation Metric:
Hit@1, Hit@3, Hit@5
"""

import os
import json
import sys
import csv
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src modules are in search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from vectorstore.embedder import Embedder
from retrieval.hybrid_retriever import HybridRetriever

# -----------------------------
# Project Root & Configuration
# -----------------------------

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "evaluation" / "evaluation.json"
OUTPUT_FILE = ROOT / "evaluation" / "benchmark_results.json"
CSV_OUTPUT_FILE = ROOT / "evaluation" / "benchmark_report.csv"
DB_PATH = str(ROOT / "data" / "vectorstore")

# -----------------------------
# Load Evaluation Questions
# -----------------------------

with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = json.load(f)

print("=" * 70)
print("UNIFIED RAG BENCHMARK EVALUATION")
print("=" * 70)
print(f"Questions Loaded : {len(questions)}")

# -----------------------------
# Load ChromaDB and Embedder
# -----------------------------

client = chromadb.PersistentClient(path=DB_PATH)
embedder = Embedder()

COLLECTIONS = {
    "fixed": client.get_collection(name="rag_fixed"),
    "semantic": client.get_collection(name="rag_semantic")
}

# -----------------------------
# Evaluation Functions
# -----------------------------

def evaluate_collection(collection):
    """
    Run all questions against a single ChromaDB collection.
    Returns Hit@1, Hit@3, Hit@5.
    """
    hit1 = hit3 = hit5 = 0
    total = len(questions)

    for question in questions:
        query_text = question["question"]
        expected_sources = set(question["expected_sources"])

        # Embed the query
        query_embedding = embedder.embed_query(query_text)

        # Query ChromaDB (retrieve top 5)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
            include=["metadatas"]
        )

        # Extract the source path from each returned chunk
        retrieved_sources = []
        if results["metadatas"] and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                source = metadata.get("source", "")
                retrieved_sources.append(source)

        # Check hits
        if expected_sources & set(retrieved_sources[:1]):
            hit1 += 1
        if expected_sources & set(retrieved_sources[:3]):
            hit3 += 1
        if expected_sources & set(retrieved_sources[:5]):
            hit5 += 1

    return {
        "hit_at_1": hit1 / total,
        "hit_at_3": hit3 / total,
        "hit_at_5": hit5 / total,
        "total_questions": total
    }

def evaluate_hybrid(use_reranker=False):
    """
    Run all questions using HybridRetriever (BM25 + Vector + RRF).
    Optionally apply Cross-Encoder Re-ranking.
    """
    retriever = HybridRetriever(strategy="semantic")
    hit1 = hit3 = hit5 = 0

    for question in questions:
        query = question["question"]
        expected_sources = set(question["expected_sources"])

        results = retriever.retrieve(
            query,
            top_k=5,
            use_reranker=use_reranker
        )

        retrieved_sources = [
            item["metadata"]["source"]
            for item in results
        ]

        h1 = bool(expected_sources & set(retrieved_sources[:1]))
        h3 = bool(expected_sources & set(retrieved_sources[:3]))
        h5 = bool(expected_sources & set(retrieved_sources[:5]))

        hit1 += h1
        hit3 += h3
        hit5 += h5

    total = len(questions)

    return {
        "hit_at_1": hit1 / total,
        "hit_at_3": hit3 / total,
        "hit_at_5": hit5 / total,
        "total_questions": total
    }

# -----------------------------
# Run Benchmark
# -----------------------------

results = {}

# 1. Run Vector-only strategies
for name, collection in COLLECTIONS.items():
    print()
    print("Running:", name.upper())
    results[name] = evaluate_collection(collection)

# 2. Run Hybrid strategy (BM25 + Vector)
print()
print("Running: HYBRID")
results["hybrid"] = evaluate_hybrid(use_reranker=False)

# 3. Run Hybrid + Reranker strategy
print()
print("Running: HYBRID + RERANKER")
results["hybrid_reranked"] = evaluate_hybrid(use_reranker=True)

# -----------------------------
# Save
# -----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# -----------------------------
# Generate CSV Report
# -----------------------------

with open(
    CSV_OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Strategy",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Total Questions"
        ]
    )

    for method, score in results.items():

        writer.writerow(
            [
                method.upper(),
                f"{score['hit_at_1']:.2%}",
                f"{score['hit_at_3']:.2%}",
                f"{score['hit_at_5']:.2%}",
                score["total_questions"]
            ]
        )

print()
print("="*70)
print("Benchmark Completed")
print("Saved:", OUTPUT_FILE)
print("Saved:", CSV_OUTPUT_FILE)
print("="*70)

# Print Summary
print("\nRESULT SUMMARY")
for method, score in results.items():
    print(
        f"""
{method.upper()}

Hit@1 : {score['hit_at_1']:.2%}
Hit@3 : {score['hit_at_3']:.2%}
Hit@5 : {score['hit_at_5']:.2%}
"""
    )
