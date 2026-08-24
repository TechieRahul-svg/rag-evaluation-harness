import chromadb


DB_PATH = "data/vectorstore"


client = chromadb.PersistentClient(
    path=DB_PATH
)


collections = client.list_collections()


print("\nVector Database Collections:\n")


for collection in collections:

    print(
        collection.name,
        "→",
        collection.count(),
        "vectors"
    )