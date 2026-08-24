import json


FILES = [
    "data/processed/fixed_chunks.json",
    "data/processed/recursive_chunks.json",
    "data/processed/semantic_chunks.json"
]


for file in FILES:

    try:

        with open(file, encoding="utf-8") as f:
            chunks = json.load(f)


        empty = 0
        small = 0
        large = 0


        for chunk in chunks:

            length = len(chunk["text"])


            if length == 0:
                empty += 1


            if length < 50:
                small += 1


            if length > 2000:
                large += 1



        print("\n", file)

        print("Total:", len(chunks))

        print("Empty:", empty)

        print("Small:", small)

        print("Large:", large)


    except FileNotFoundError:

        print(file, "not found")