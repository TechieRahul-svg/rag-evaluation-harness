import os
from pathlib import Path
import json
import chromadb

from embedder import Embedder

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(ROOT / "data" / "vectorstore")


DATASETS = {
    "rag_fixed": str(ROOT / "data" / "processed" / "fixed_chunks.json"),
    "rag_recursive": str(ROOT / "data" / "processed" / "recursive_chunks.json"),
    "rag_semantic": str(ROOT / "data" / "processed" / "semantic_chunks.json")
}


# Initialize ChromaDB
client = chromadb.PersistentClient(
    path=DB_PATH
)


# Load embedding model only once
embedder = Embedder()


def build_collection(collection_name, input_file):

    print("\n" + "=" * 60)
    print("Building:", collection_name)
    print("=" * 60)

    # Load chunks
    with open(
        input_file,
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    print("Loaded chunks:", len(chunks))


    # Delete old collection if it exists
    try:

        client.delete_collection(
            name=collection_name
        )

        print("Old collection deleted")

    except Exception:

        pass


    # Create fresh collection
    collection = client.create_collection(
        name=collection_name
    )


    ids = []
    texts = []
    metadatas = []


    for chunk in chunks:

        chunk_id = str(
            chunk["chunk_id"]
        )

        ids.append(chunk_id)

        texts.append(
            chunk["text"]
        )


        metadata = chunk["metadata"].copy()

        metadata["chunk_id"] = chunk_id

        metadatas.append(metadata)


    print("Generating embeddings...")


    embeddings = embedder.embed_documents(
        texts
    )


    print("Storing vectors...")


    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


    print(
        "Collection:",
        collection_name
    )

    print(
        "Vectors:",
        collection.count()
    )



for collection_name, input_file in DATASETS.items():

    build_collection(
        collection_name,
        input_file
    )


print("\nAll vector stores created successfully.")