"""
run_reranked_evaluation.py
--------------------------
Two-stage retrieval with cross-encoder re-ranking.

Stage 1: Retrieve top-CANDIDATE_K candidates from ChromaDB (bi-encoder).
Stage 2: Re-score each candidate against the query using a cross-encoder.
         Keep the top TOP_K after re-ranking.

Prints Hit@1/3/5 before and after re-ranking side-by-side for all 3 strategies.
Saves results to evaluation/reranked_results.json.

Run from the project root:
    python src/retrieval/run_reranked_evaluation.py
"""

import json
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "vectorstore"),
)

import chromadb
from embedder import Embedder
from sentence_transformers import CrossEncoder

# --------------------------------------------------
# Config
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "evaluation" / "evaluation.json"
OUTPUT_FILE    = ROOT / "evaluation" / "reranked_results.json"
DB_PATH        = str(ROOT / "data" / "vectorstore")

COLLECTIONS = {
    "fixed":     "rag_fixed",
    "recursive": "rag_recursive",
    "semantic":  "rag_semantic",
}

CANDIDATE_K = 20   # how many to retrieve from ChromaDB before re-ranking
TOP_K       = 5    # how many to keep after re-ranking

# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading bi-encoder (all-MiniLM-L6-v2)...")
embedder = Embedder()

print("Loading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --------------------------------------------------
# Load questions and ChromaDB
# --------------------------------------------------

with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = json.load(f)

client = chromadb.PersistentClient(path=DB_PATH)

# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_with_reranking(collection_name, questions):
    collection = client.get_collection(name=collection_name)

    before_hits1 = before_hits3 = before_hits5 = 0
    after_hits1  = after_hits3  = after_hits5  = 0
    per_question = []

    for question in questions:
        qid            = question["id"]
        query_text     = question["question"]
        expected_sources = set(question["expected_sources"])

        # --- Stage 1: bi-encoder retrieval ---
        query_embedding = embedder.embed_query(query_text)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=CANDIDATE_K,
            include=["metadatas", "documents", "distances"],
        )

        candidates_meta = results["metadatas"][0]
        candidates_docs = results["documents"][0]

        # Sources before re-ranking (top-K of bi-encoder)
        sources_before = [m.get("source", "") for m in candidates_meta[:TOP_K]]

        # --- Stage 2: cross-encoder re-ranking ---
        pairs = [(query_text, doc) for doc in candidates_docs]
        scores = cross_encoder.predict(pairs)

        # Sort candidates by cross-encoder score (descending)
        ranked = sorted(
            zip(scores, candidates_meta, candidates_docs),
            key=lambda x: x[0],
            reverse=True,
        )

        sources_after = [meta.get("source", "") for _, meta, _ in ranked[:TOP_K]]

        # --- Before hits ---
        b1 = bool(expected_sources & set(sources_before[:1]))
        b3 = bool(expected_sources & set(sources_before[:3]))
        b5 = bool(expected_sources & set(sources_before[:5]))

        if b1: before_hits1 += 1
        if b3: before_hits3 += 1
        if b5: before_hits5 += 1

        # --- After hits ---
        a1 = bool(expected_sources & set(sources_after[:1]))
        a3 = bool(expected_sources & set(sources_after[:3]))
        a5 = bool(expected_sources & set(sources_after[:5]))

        if a1: after_hits1 += 1
        if a3: after_hits3 += 1
        if a5: after_hits5 += 1

        per_question.append({
            "question_id":      qid,
            "question":         query_text,
            "expected_sources": list(expected_sources),
            "sources_before":   sources_before,
            "sources_after":    sources_after,
            "before_hit_at_1":  b1,
            "before_hit_at_3":  b3,
            "before_hit_at_5":  b5,
            "after_hit_at_1":   a1,
            "after_hit_at_3":   a3,
            "after_hit_at_5":   a5,
        })

    total = len(questions)
    return {
        "before": {
            "hit_at_1": before_hits1 / total,
            "hit_at_3": before_hits3 / total,
            "hit_at_5": before_hits5 / total,
        },
        "after": {
            "hit_at_1": after_hits1 / total,
            "hit_at_3": after_hits3 / total,
            "hit_at_5": after_hits5 / total,
        },
        "per_question": per_question,
    }

# --------------------------------------------------
# Run across all strategies
# --------------------------------------------------

print("=" * 70)
print("RAG RETRIEVAL EVALUATION WITH CROSS-ENCODER RE-RANKING")
print("=" * 70)
print(f"Questions   : {len(questions)}")
print(f"Candidate K : {CANDIDATE_K}  (bi-encoder retrieval)")
print(f"Top K       : {TOP_K}  (after re-ranking)")
print()

all_results = {}

for strategy, collection_name in COLLECTIONS.items():

    print(f"\n{'=' * 70}")
    print(f"Strategy: {strategy.upper()}")
    print("=" * 70)

    result = evaluate_with_reranking(collection_name, questions)
    all_results[strategy] = result

    b = result["before"]
    a = result["after"]

    print(f"\n  {'Metric':<10} {'Before':>8}  {'After':>8}  {'Delta':>8}")
    print("  " + "-" * 40)
    print(f"  {'Hit@1':<10} {b['hit_at_1']:>8.2%}  {a['hit_at_1']:>8.2%}  {a['hit_at_1']-b['hit_at_1']:>+8.2%}")
    print(f"  {'Hit@3':<10} {b['hit_at_3']:>8.2%}  {a['hit_at_3']:>8.2%}  {a['hit_at_3']-b['hit_at_3']:>+8.2%}")
    print(f"  {'Hit@5':<10} {b['hit_at_5']:>8.2%}  {a['hit_at_5']:>8.2%}  {a['hit_at_5']-b['hit_at_5']:>+8.2%}")

    print()
    print(f"  {'Q':<5} {'Before H@1':>11}  {'After H@1':>10}  Question")
    print("  " + "-" * 70)
    for q in result["per_question"]:
        b1 = "YES" if q["before_hit_at_1"] else "---"
        a1 = "YES" if q["after_hit_at_1"]  else "---"
        improved = " <-- FIXED" if a1 == "YES" and b1 == "---" else ""
        regressed = " <-- BROKE" if a1 == "---" and b1 == "YES" else ""
        print(f"  Q{q['question_id']:<4} {b1:>11}  {a1:>10}  {q['question'][:45]}{improved}{regressed}")

# --------------------------------------------------
# Save
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"\n{'=' * 70}")
print("Re-ranking evaluation complete.")
print(f"Results saved: {OUTPUT_FILE}")
print("=" * 70)
