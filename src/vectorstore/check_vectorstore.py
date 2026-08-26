import os
from pathlib import Path
import chromadb

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(ROOT / "data" / "vectorstore")


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