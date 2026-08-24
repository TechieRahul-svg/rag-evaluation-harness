import json
from collections import Counter


with open(
    "data/processed/documents.json",
    encoding="utf-8"
) as file:

    documents = json.load(file)



categories = []


for doc in documents:

    categories.append(
        doc["metadata"]["category"]
    )


print(
    "Total documents:",
    len(documents)
)


print(
    "\nCategories:"
)


for category, count in Counter(categories).items():

    print(
        category,
        ":",
        count
    )