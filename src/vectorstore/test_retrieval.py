import chromadb

from embedder import Embedder


DB_PATH = "data/vectorstore"

COLLECTION_NAME = "rag_semantic"


# Initialize ChromaDB

client = chromadb.PersistentClient(
    path=DB_PATH
)


collection = client.get_collection(
    name=COLLECTION_NAME
)


# Initialize embedding model

embedder = Embedder()


# Test question

question = (
    "How does dependency injection "
    "work in FastAPI?"
)


# Convert question into embedding

query_embedding = embedder.embed_query(
    question
)


# Search ChromaDB

results = collection.query(

    query_embeddings=[
        query_embedding.tolist()
    ],

    n_results=5
)


print("\nQuestion:")
print(question)


print("\nRetrieved Documents:\n")


for i in range(5):

    print("=" * 70)

    print(
        f"Rank: {i + 1}"
    )


    print(
        "Source:",
        results["metadatas"][0][i]["source"]
    )


    print(
        "Category:",
        results["metadatas"][0][i]["category"]
    )


    print(
        "Distance:",
        results["distances"][0][i]
    )


    print("\nContent:")

    print(
        results["documents"][0][i][:500]
    )