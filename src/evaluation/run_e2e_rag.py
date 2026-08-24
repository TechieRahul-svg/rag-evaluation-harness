import json
import os
import sys
import argparse
from pathlib import Path

# Ensure src modules are in search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pipeline.rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Run End-to-End RAG evaluation on the benchmark questions.")
    parser.add_argument("--strategy", type=str, default="semantic", choices=["fixed", "recursive", "semantic"],
                        help="Chunking strategy to use (default: semantic)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder re-ranking")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable BM25 + Vector hybrid search")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process for speed")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    QUESTIONS_FILE = ROOT / "evaluation" / "evaluation.json"
    OUTPUT_FILE = ROOT / "evaluation" / f"e2e_results_{args.strategy}.json"

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[:args.limit]

    print("=" * 70)
    print("END-TO-END HYBRID RAG PIPELINE RUNNER & EVALUATION")
    print("=" * 70)
    print(f"Total questions to run: {len(questions)}")
    print(f"Strategy:               {args.strategy.upper()}")
    print(f"Hybrid Search active:   {not args.no_hybrid}")
    print(f"Re-ranking active:      {not args.no_rerank}")
    print("=" * 70 + "\n")

    pipeline = RAGPipeline()
    results = []

    for idx, q in enumerate(questions, 1):
        print(f"Processing Q{q['id']} ({idx}/{len(questions)}): {q['question']}")
        
        # Run query through RAG pipeline
        res = pipeline.query(
            query_text=q["question"],
            strategy=args.strategy,
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_rerank
        )
        
        # Format comparison
        print("-" * 50)
        print("RETRIEVED SOURCES:")
        print(res["retrieved_sources"])
        print("\nEXPECTED ANSWER:")
        print(q["expected_answer"][:200] + "...")
        print("\nGENERATED ANSWER:")
        print(res["answer"])
        print("-" * 50 + "\n")

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "expected_sources": q["expected_sources"],
            "expected_answer": q["expected_answer"],
            "retrieved_sources": res["retrieved_sources"],
            "generated_answer": res["answer"],
            "retrieved_chunks": res["retrieved_chunks"]
        })

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
