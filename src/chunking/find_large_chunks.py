import json


FILE = "data/processed/semantic_chunks.json"


with open(FILE, encoding="utf-8") as f:
    chunks = json.load(f)



for chunk in chunks:

    length = len(chunk["text"])

    if length > 2000:

        print("\nLength:", length)

        print(
            "Source:",
            chunk["metadata"]["source"]
        )

        print(
            chunk["text"][:300]
        )

        print("-"*50)