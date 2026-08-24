import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = "data/processed/documents.json"

OUTPUT_FILE = "data/processed/recursive_chunks.json"



# Load documents

with open(
    INPUT_FILE,
    encoding="utf-8"
) as f:

    documents = json.load(f)



# Recursive splitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=50,

    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)



chunks = []

chunk_id = 1



for doc in documents:


    split_texts = splitter.split_text(
        doc["text"]
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



# Save output

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



print("Recursive Chunking Completed")

print("Total documents:", len(documents))

print("Total chunks:", len(chunks))

print("Saved:", OUTPUT_FILE)