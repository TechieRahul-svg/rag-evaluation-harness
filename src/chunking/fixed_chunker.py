import json


INPUT_FILE = "data/processed/documents.json"

OUTPUT_FILE = "data/processed/fixed_chunks.json"


def fixed_split(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

        # Avoid infinite loop
        if start >= len(text):
            break

    return chunks



# Load processed documents

with open(
    INPUT_FILE,
    encoding="utf-8"
) as f:

    documents = json.load(f)



chunks = []

chunk_id = 1



# Apply fixed chunking

for doc in documents:

    split_texts = fixed_split(
        doc["text"],
        chunk_size=500,
        overlap=50
    )


    for text in split_texts:

        chunks.append(
            {
                "chunk_id": chunk_id,

                "text": text,

                "metadata": doc["metadata"]
            }
        )

        chunk_id += 1



# Save chunks

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        indent=4,
        ensure_ascii=False
    )



print("Fixed Chunking Completed")
print("Total documents:", len(documents))
print("Total chunks:", len(chunks))
print("Saved:", OUTPUT_FILE)