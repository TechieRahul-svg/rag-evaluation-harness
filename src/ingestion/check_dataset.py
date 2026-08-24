import json


with open(
    "data/processed/documents.json",
    encoding="utf-8"
) as file:

    docs = json.load(file)


empty = []

for doc in docs:

    if len(doc["text"].strip()) < 50:
        empty.append(
            doc["metadata"]["source"]
        )


print(
    "Total documents:",
    len(docs)
)

print(
    "Empty documents:",
    len(empty)
)


for item in empty:
    print(item)