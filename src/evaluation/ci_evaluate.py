
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_FILE = Path("evaluation/retrieval_results.json")

# Minimum acceptable performance
MINIMUM_HIT5 = 0.90
EXPECTED_QUESTIONS = 50
REQUIRED_STRATEGIES = {"fixed", "recursive", "semantic"}


def main():
    print("=" * 60)
    print("RAG CI QUALITY CHECK")
    print("=" * 60)

    # 1. Check that the results file exists.
    if not RESULT_FILE.is_file():
        print(f"❌ Results file not found: {RESULT_FILE}")
        return 1

    # 2. Check that the JSON can be parsed.
    try:
        with RESULT_FILE.open(encoding="utf-8") as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"❌ Could not read valid JSON results: {exc}")
        return 1

    # 3. Validate the expected strategies.
    if not isinstance(results, dict):
        print("❌ Results must be a JSON object.")
        return 1

    missing = REQUIRED_STRATEGIES - results.keys()
    if missing:
        print(f"❌ Missing strategies: {', '.join(sorted(missing))}")
        return 1

    strategy = "recursive"
    strategy_results = results[strategy]

    if not isinstance(strategy_results, dict):
        print(f"❌ Results for '{strategy}' must be an object.")
        return 1

    # 4. Validate the question count.
    total_questions = strategy_results.get("total_questions")
    if (
        isinstance(total_questions, bool)
        or not isinstance(total_questions, int)
        or total_questions != EXPECTED_QUESTIONS
    ):
        print(
            f"❌ Expected {EXPECTED_QUESTIONS} questions for "
            f"'{strategy}', got {total_questions!r}."
        )
        return 1

    # 5. Validate Hit@5.
    hit5 = strategy_results.get("hit_at_5")
    if (
        isinstance(hit5, bool)
        or not isinstance(hit5, (int, float))
        or not 0 <= hit5 <= 1
    ):
        print(f"❌ Invalid Hit@5 value: {hit5!r}")
        return 1

    print("Strategy:", strategy)
    print("Questions:", total_questions)
    print("Hit@5:", hit5)
    print("Required:", MINIMUM_HIT5)

    # 6. Apply the existing quality threshold.
    if hit5 >= MINIMUM_HIT5:
        print("\n✅ RAG Evaluation Passed")
        return 0

    print("\n❌ RAG Evaluation Failed")
    print("Retrieval quality dropped below threshold.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
