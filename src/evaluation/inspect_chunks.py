"""
inspect_chunks.py
-----------------
Diagnose retrieval failures for Q6 (path parameters) and Q8 (response model).

For each failing question we:
  1. Run the query against all 3 ChromaDB collections and show the full
     top-5 ranked results with distances so we can see *why* the wrong
     documents win.
  2. Dump every chunk that belongs to the expected source file so we can
     see how (or whether) that file was split and indexed.

Run from the project root:
    python src/evaluation/inspect_chunks.py
"""

import json
import sys
import os
import textwrap
from pathlib import Path

# UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Make the vectorstore package importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "vectorstore"),
)

import chromadb
from embedder import Embedder

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(ROOT / "data" / "vectorstore")

COLLECTIONS = {
    "fixed":     "rag_fixed",
    "recursive": "rag_recursive",
    "semantic":  "rag_semantic",
}

# Chunk JSON files (used for the raw-chunk dump)
CHUNK_FILES = {
    "fixed":     ROOT / "data" / "processed" / "fixed_chunks.json",
    "recursive": ROOT / "data" / "processed" / "recursive_chunks.json",
    "semantic":  ROOT / "data" / "processed" / "semantic_chunks.json",
}

TOP_K = 5
CHUNK_PREVIEW = 400   # characters to show per chunk

# --------------------------------------------------
# Failing questions to diagnose
# --------------------------------------------------

FAILING = [
    {
        "id": 17,
        "question": "How do you read HTTP header parameters in FastAPI using Header(), and how does FastAPI handle underscore to hyphen conversion?",
        "expected_sources": [
            "tutorial/header-params.md",
        ],
    },
    {
        "id": 20,
        "question": "How do sub-dependencies work in FastAPI when one Depends() function itself declares another Depends() parameter?",
        "expected_sources": [
            "tutorial/dependencies/sub-dependencies.md",
        ],
    },
]

# --------------------------------------------------
# Setup
# --------------------------------------------------

client = chromadb.PersistentClient(path=DB_PATH)
embedder = Embedder()


def divider(char="=", width=80):
    print(char * width)


def section(title):
    divider()
    print(title)
    divider()


# --------------------------------------------------
# Part 1 — Query each collection and show ranked results
# --------------------------------------------------

def inspect_retrieval(question_id, query_text, expected_sources):
    section(f"RETRIEVAL DIAGNOSIS  |  Q{question_id}: {query_text}")
    print(f"Expected sources : {expected_sources}\n")

    query_embedding = embedder.embed_query(query_text)

    for strategy, collection_name in COLLECTIONS.items():
        collection = client.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K,
            include=["metadatas", "documents", "distances"],
        )

        metadatas  = results["metadatas"][0]
        documents  = results["documents"][0]
        distances  = results["distances"][0]

        print(f"\n  [{strategy.upper()}]  (lower distance = more similar)")
        print(f"  {'Rank':<5} {'Distance':>9}  {'Hit':^4}  Source")
        print("  " + "-" * 70)

        for rank, (meta, doc, dist) in enumerate(
            zip(metadatas, documents, distances), start=1
        ):
            source = meta.get("source", "<unknown>")
            hit = "YES" if source in expected_sources else "---"
            print(f"  {rank:<5} {dist:>9.4f}  {hit}    {source}")

            # If it's a hit show the first line of the chunk text
            if source in expected_sources:
                first_line = doc.strip().splitlines()[0][:80]
                print(f"        >> \"{first_line}\"")

        print()


# --------------------------------------------------
# Part 2 — Dump all chunks from the expected source files
# --------------------------------------------------

def dump_chunks_for_source(question_id, expected_sources):
    for strategy, chunk_file in CHUNK_FILES.items():

        print(f"\n  [{strategy.upper()}]  chunks from expected source(s)")
        print("  " + "-" * 70)

        with open(chunk_file, encoding="utf-8") as f:
            chunks = json.load(f)

        # Filter to chunks whose source matches any expected source
        matching = [
            c for c in chunks
            if c.get("metadata", {}).get("source", "") in expected_sources
        ]

        if not matching:
            print("  WARNING: NO CHUNKS FOUND for this source in this strategy!")
            print(
                "  This means the file was never ingested, or the "
                "source path in metadata differs from expected."
            )
            # Try a loose match to help debug
            loose = [
                c for c in chunks
                if any(
                    s.split("/")[-1] in c.get("metadata", {}).get("source", "")
                    for s in expected_sources
                )
            ]
            if loose:
                print(f"  Loose match found {len(loose)} chunk(s) -- sample sources:")
                seen = set()
                for c in loose:
                    src = c.get("metadata", {}).get("source", "?")
                    if src not in seen:
                        print(f"    - {src}")
                        seen.add(src)
            continue

        print(f"  Found {len(matching)} chunk(s)\n")

        for i, chunk in enumerate(matching, 1):
            meta   = chunk.get("metadata", {})
            source = meta.get("source", "?")
            cid    = chunk.get("chunk_id", "?")
            text   = chunk.get("text", "")
            words  = len(text.split())

            print(f"  Chunk {i}/{len(matching)}  |  id={cid}  |  source={source}  |  words={words}")

            # Show a preview of the chunk text
            preview = text[:CHUNK_PREVIEW].strip().replace("\n", " ")
            preview = textwrap.fill(
                preview, width=75,
                initial_indent="    ",
                subsequent_indent="    "
            )
            print(preview)
            if len(text) > CHUNK_PREVIEW:
                print(f"    ... [{len(text) - CHUNK_PREVIEW} more chars]")
            print()


# --------------------------------------------------
# Main
# --------------------------------------------------

print("\n")
section("CHUNK INSPECTION -- Diagnosing retrieval failures")
print("Questions under investigation: Q6 (path-params) and Q8 (response-model)\n")

for item in FAILING:

    # --- Retrieval diagnosis ---
    inspect_retrieval(
        question_id=item["id"],
        query_text=item["question"],
        expected_sources=set(item["expected_sources"]),
    )

    # --- Raw chunk dump ---
    divider("-")
    print(f"RAW CHUNKS  |  Q{item['id']}  |  Expected sources: {item['expected_sources']}")
    divider("-")
    dump_chunks_for_source(item["id"], set(item["expected_sources"]))

    print("\n")

divider()
print("Inspection complete.")
divider()
