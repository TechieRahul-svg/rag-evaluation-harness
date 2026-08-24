import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FILE = "evaluation/retrieval_results.json"

with open(FILE, encoding="utf-8") as f:
    results = json.load(f)


# --------------------------------------------------
# Show all failures (questions that missed Hit@5)
# across every strategy
# --------------------------------------------------

for strategy, data in results.items():

    failures = [
        q for q in data["per_question"]
        if not q["hit_at_5"]
    ]

    print("\n" + "=" * 80)
    print(f"STRATEGY: {strategy.upper()}")
    print(f"Failures (missed Hit@5): {len(failures)}")
    print("=" * 80)

    for result in failures:

        print(f"\n  Question ID : {result['question_id']}")
        print(f"  Question    : {result['question']}")
        print(f"  Expected    : {result['expected_sources']}")
        print(f"  Retrieved   :")

        for rank, source in enumerate(
            result["retrieved_sources"], start=1
        ):
            print(f"    {rank}. {source}")


# --------------------------------------------------
# Cross-strategy summary — questions that failed
# in ALL three strategies
# --------------------------------------------------

print("\n" + "=" * 80)
print("QUESTIONS THAT FAILED IN ALL 3 STRATEGIES")
print("=" * 80)

strategies = list(results.keys())

# Build a dict: question_id -> list of hit_at_5 per strategy
all_questions = results[strategies[0]]["per_question"]

for q in all_questions:
    qid = q["question_id"]
    all_failed = all(
        not results[s]["per_question"][qid - 1]["hit_at_5"]
        for s in strategies
    )

    if all_failed:
        print(f"\n  Q{qid}: {q['question']}")
        print(f"  Expected: {q['expected_sources']}")
        print("  Retrieved per strategy:")

        for s in strategies:
            retrieved = results[s]["per_question"][qid - 1]["retrieved_sources"]
            print(f"    [{s.upper()}]")
            for rank, src in enumerate(retrieved, 1):
                print(f"      {rank}. {src}")
