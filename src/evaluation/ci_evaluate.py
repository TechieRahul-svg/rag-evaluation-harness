import json
import sys


RESULT_FILE = "evaluation/retrieval_results.json"

# Minimum acceptable performance
MINIMUM_HIT5 = 0.90


def main():

    with open(
        RESULT_FILE,
        encoding="utf-8"
    ) as f:
        results = json.load(f)


    # Change this according to your best retriever
    strategy = "recursive"


    hit5 = results[strategy]["hit_at_5"]


    print("=" * 60)
    print("RAG CI QUALITY CHECK")
    print("=" * 60)

    print("Strategy:", strategy)
    print("Hit@5:", hit5)
    print("Required:", MINIMUM_HIT5)


    if hit5 >= MINIMUM_HIT5:

        print("\n✅ RAG Evaluation Passed")

        sys.exit(0)


    else:

        print("\n❌ RAG Evaluation Failed")

        print(
            "Retrieval quality dropped below threshold"
        )

        sys.exit(1)



if __name__ == "__main__":
    main()