import sys
import os
import chromadb

# Make the vectorstore package importable regardless of CWD
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "vectorstore")
)

from embedder import Embedder


DB_PATH = "data/vectorstore"


COLLECTIONS = [
    "rag_fixed",
    "rag_recursive",
    "rag_semantic"
]


QUESTION = (
    "How does dependency injection work in FastAPI?"
)


TOP_K = 5


# Initialize ChromaDB

client = chromadb.PersistentClient(
    path=DB_PATH
)


# Initialize embedding model

embedder = Embedder()


# Embed question

query_embedding = embedder.embed_query(
    QUESTION
)


print("\n" + "=" * 80)

print("QUESTION")

print("=" * 80)

print(QUESTION)


# Test every retrieval strategy

for collection_name in COLLECTIONS:

    print("\n\n" + "#" * 80)

    print(
        f"STRATEGY: {collection_name}"
    )

    print("#" * 80)


    collection = client.get_collection(
        name=collection_name
    )


    results = collection.query(

        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=TOP_K
    )


    for i in range(TOP_K):

        metadata = (
            results["metadatas"][0][i]
        )

        distance = (
            results["distances"][0][i]
        )

        document = (
            results["documents"][0][i]
        )


        print("\n" + "-" * 70)

        print(
            f"Rank: {i + 1}"
        )

        print(
            "Source:",
            metadata["source"]
        )

        print(
            "Category:",
            metadata["category"]
        )

        print(
            "Distance:",
            round(distance, 4)
        )

        print("\nContent:")

        print(
            document[:500]
        )