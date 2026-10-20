import json
import os
import re
import sys
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectorstore.embedder import Embedder
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, strategy="semantic", db_path="data/vectorstore", k_constant=60):
        print(f"Initializing HybridRetriever for strategy '{strategy}'...")
        self.strategy = strategy
        self.root = Path(__file__).resolve().parents[2]
        
        # Ensure db_path is absolute relative to project root
        if not os.path.isabs(db_path):
            self.db_path = str(self.root / db_path)
        else:
            self.db_path = db_path
            
        self.k_constant = k_constant
        
        # Load ChromaDB VectorStore
        self.client = chromadb.PersistentClient(path=self.db_path)
        collection_names = {
            "fixed": "rag_fixed",
            "recursive": "rag_recursive",
            "semantic": "rag_semantic"
        }
        self.collection_name = collection_names[strategy]
        self.collection = self.client.get_collection(name=self.collection_name)
        
        # Load raw chunk data for BM25 with absolute paths
        chunk_files = {
            "fixed": str(self.root / "data" / "processed" / "fixed_chunks.json"),
            "recursive": str(self.root / "data" / "processed" / "recursive_chunks.json"),
            "semantic": str(self.root / "data" / "processed" / "semantic_chunks.json")
        }
        self.chunk_file = chunk_files[strategy]
        with open(self.chunk_file, encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        # Initialize BM25
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Lazy loaded cross encoder
        self.cross_encoder = None
        self.embedder = Embedder()
        print("HybridRetriever initialized successfully.")

    def _tokenize(self, text):
        # Basic word tokenization (lowercase, strip non-alphanumeric)
        text = text.lower()
        words = re.findall(r"\b[a-z0-9_\-]+\b", text)
        return words

    def _get_cross_encoder(self):
        if self.cross_encoder is None:
            print("Loading cross-encoder for hybrid re-ranking...")
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self.cross_encoder

    def retrieve(self, query_text: str, top_k=5, candidate_k=20, use_reranker=False):
        """
        Retrieves top_k chunks using Hybrid (BM25 + Vector) search fused with RRF.
        """
        # --- 1. BM25 Retrieval ---
        tokenized_query = self._tokenize(query_text)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort chunks by BM25 score descending
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:candidate_k]
        
        bm25_results = []
        for rank, idx in enumerate(bm25_ranked_indices):
            chunk = self.chunks[idx]
            bm25_results.append((str(chunk["chunk_id"]), rank + 1))

        # --- 2. Vector Retrieval ---
        query_embedding = self.embedder.embed_query(query_text)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=candidate_k,
            include=["metadatas", "documents"]
        )
        
        vec_ids = vector_results["ids"][0] if vector_results["ids"] else []
        vector_ranked = [(str(vid), rank + 1) for rank, vid in enumerate(vec_ids)]

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        # RRF score: Sum of 1 / (k + rank) across retrievers
        rrf_scores = {}
        
        # Map chunk IDs to actual documents/metadatas
        chunk_map = {str(c["chunk_id"]): c for c in self.chunks}
        
        # Process BM25 ranks
        for cid, rank in bm25_results:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_constant + rank))
            
        # Process Vector ranks
        for cid, rank in vector_ranked:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_constant + rank))

        # Sort all candidates by RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]

        # Fetch full chunk details
        retrieved = []
        for cid, rrf_score in sorted_candidates:
            chunk = chunk_map.get(cid)
            if chunk:
                retrieved.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "rrf_score": rrf_score
                })

        # --- 4. Cross-Encoder Re-ranking ---
        if use_reranker and retrieved:
            cross_encoder = self._get_cross_encoder()
            pairs = [(query_text, item["text"]) for item in retrieved]
            rerank_scores = cross_encoder.predict(pairs)
            
            # Re-sort by cross-encoder score descending
            for item, score in zip(retrieved, rerank_scores):
                item["rerank_score"] = float(score)
                
            retrieved = sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)

        return retrieved[:top_k]

if __name__ == "__main__":
    # Test hybrid retrieval
    retriever = HybridRetriever(strategy="semantic")
    results = retriever.retrieve(
        "How do you declare HTTP header parameters in FastAPI using Header()?",
        top_k=3,
        use_reranker=False
    )
    print("\n" + "="*50)
    print("HYBRID SEARCH TEST RESULTS:")
    for rank, item in enumerate(results, 1):
        print(f"\n{rank}. Chunk ID: {item['chunk_id']} (Source: {item['metadata']['source']})")
        print(f"   RRF Score: {item['rrf_score']:.6f} | Rerank Score: {item.get('rerank_score', 0.0):.4f}")
        print(f"   Text: {item['text'][:120]}...")
    print("="*50)
