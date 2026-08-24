import json
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "data/processed/documents.json"
OUTPUT_FILE = "data/processed/semantic_chunks.json"

model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]{0,200}\}", " ", text)
    text = re.sub(r"\S+docssrc\S*", " ", text)
    text = re.sub(r"hl\[\d+[:\d]*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+|\n{1,}", text)
    return [s.strip() for s in parts if len(s.strip()) > 40]


def semantic_chunk(sentences, threshold=0.5, min_chunk_size=400, max_chunk_size=1200):
    if not sentences:
        return []
    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = []
    current_length = 0
    for i, sentence in enumerate(sentences):
        should_split = False
        if i > 0:
            similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            if similarity < threshold and current_length >= min_chunk_size:
                should_split = True
            if current_length >= max_chunk_size:
                should_split = True
        if should_split:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


with open(INPUT_FILE, encoding="utf-8") as f:
    documents = json.load(f)

chunks = []
chunk_id = 1

for doc in documents:
    sentences = split_sentences(doc["text"])
    semantic_chunks = semantic_chunk(sentences)
    for chunk in semantic_chunks:
        chunks.append({"chunk_id": chunk_id, "text": chunk, "metadata": doc["metadata"]})
        chunk_id += 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)

print("Semantic Chunking Completed")
print("Total documents:", len(documents))
print("Total chunks:", len(chunks))
print("Saved:", OUTPUT_FILE)
