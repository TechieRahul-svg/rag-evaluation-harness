import json
import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Make the vectorstore package importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "vectorstore")
)

import chromadb
from embedder import Embedder

# --------------------------------------------------
# Paths and config
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "evaluation" / "evaluation.json"
OUTPUT_FILE = ROOT / "evaluation" / "retrieval_results.json"
DB_PATH = str(ROOT / "data" / "vectorstore")

COLLECTIONS = {
    "fixed": "rag_fixed",
    "recursive": "rag_recursive",
    "semantic": "rag_semantic",
}

TOP_K = 5

# --------------------------------------------------
# Load questions
# --------------------------------------------------

with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = json.load(f)

# --------------------------------------------------
# Load ChromaDB and embedder
# --------------------------------------------------

client = chromadb.PersistentClient(path=DB_PATH)
embedder = Embedder()


print("=" * 70)
print("RAG RETRIEVAL EVALUATION")
print("=" * 70)
print(f"Questions: {len(questions)}")


def evaluate_collection(collection_name, questions):
    """
    Run all questions against a single ChromaDB collection.
    Returns per-question results and aggregate Hit@K scores.
    """
    collection = client.get_collection(name=collection_name)
    per_question = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    for question in questions:
        question_id = question["id"]
        query_text = question["question"]
        expected_sources = set(question["expected_sources"])

        # Embed the question
        query_embedding = embedder.embed_query(query_text)

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K,
            include=["metadatas", "distances"]
        )

        # Extract the source path from each returned chunk
        retrieved_sources = []
        for metadata in results["metadatas"][0]:
            source = metadata.get("source", "")
            retrieved_sources.append(source)

        # Check hits
        hit_at_1 = bool(
            expected_sources & set(retrieved_sources[:1])
        )
        hit_at_3 = bool(
            expected_sources & set(retrieved_sources[:3])
        )
        hit_at_5 = bool(
            expected_sources & set(retrieved_sources[:5])
        )

        if hit_at_1:
            hits_at_1 += 1
        if hit_at_3:
            hits_at_3 += 1
        if hit_at_5:
            hits_at_5 += 1

        per_question.append({
            "question_id": question_id,
            "question": query_text,
            "expected_sources": list(expected_sources),
            "retrieved_sources": retrieved_sources,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
        })

    total = len(questions)
    return {
        "hit_at_1": hits_at_1 / total,
        "hit_at_3": hits_at_3 / total,
        "hit_at_5": hits_at_5 / total,
        "hits_at_1_count": hits_at_1,
        "hits_at_3_count": hits_at_3,
        "hits_at_5_count": hits_at_5,
        "total_questions": total,
        "per_question": per_question,
    }


# --------------------------------------------------
# Run evaluation across all strategies
# --------------------------------------------------

all_results = {}

for strategy, collection_name in COLLECTIONS.items():

    print(f"\n{'=' * 70}")
    print(f"Evaluating: {strategy.upper()}")
    print("=" * 70)

    result = evaluate_collection(collection_name, questions)
    all_results[strategy] = result

    print(f"Hit@1:  {result['hits_at_1_count']}/{result['total_questions']}  ({result['hit_at_1']:.2%})")
    print(f"Hit@3:  {result['hits_at_3_count']}/{result['total_questions']}  ({result['hit_at_3']:.2%})")
    print(f"Hit@5:  {result['hits_at_5_count']}/{result['total_questions']}  ({result['hit_at_5']:.2%})")

    # Per-question breakdown
    print()
    for q in result["per_question"]:
        h1 = "✅" if q["hit_at_1"] else "❌"
        h3 = "✅" if q["hit_at_3"] else "❌"
        h5 = "✅" if q["hit_at_5"] else "❌"
        print(
            f"  Q{q['question_id']:>2}  Hit@1:{h1}  Hit@3:{h3}  Hit@5:{h5}"
            f"  | {q['question'][:55]}"
        )

# --------------------------------------------------
# Save results
# --------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"\n{'=' * 70}")
print("Evaluation completed.")
print(f"Results saved: {OUTPUT_FILE}")
print("=" * 70)