import os
import sys
from pathlib import Path

# Ensure src modules are in search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.hybrid_retriever import HybridRetriever
from generator.groq_generator import GroqGenerator

class RAGPipeline:
    def __init__(self, db_path="data/vectorstore", model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print("Initializing unified RAGPipeline...")
        self.db_path = db_path
        self.model_name = model_name
        self.generator = GroqGenerator(fallback_model=model_name)
        
        # Cache retrievers for each strategy to avoid rebuilding BM25 indexes constantly
        self.retrievers = {}

    def _get_retriever(self, strategy: str) -> HybridRetriever:
        if strategy not in self.retrievers:
            self.retrievers[strategy] = HybridRetriever(
                strategy=strategy,
                db_path=self.db_path
            )
        return self.retrievers[strategy]

    def query(self, query_text: str, strategy: str = "semantic", use_hybrid: bool = True, use_reranker: bool = True, top_k: int = 5) -> dict:
        """
        Unified end-to-end RAG entry point:
        - Retrieves candidates using Vector, BM25, or Hybrid search.
        - Re-ranks candidates using Cross-Encoder.
        - Generates response using Groq Llama 3.3 (or local Qwen fallback).
        """
        retriever = self._get_retriever(strategy)
        
        if use_hybrid:
            # BM25 + Vector fused with RRF
            retrieved_chunks = retriever.retrieve(
                query_text=query_text,
                top_k=top_k,
                use_reranker=use_reranker
            )
        else:
            # Vector-only fallback (retrieves using the same ranker but sets candidate_k directly)
            # Since retrieval RRF score handles ranks, we can bypass RRF scoring by simulating Vector-only retrieval
            query_embedding = retriever.embedder.embed_query(query_text)
            results = retriever.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=20 if use_reranker else top_k,
                include=["metadatas", "documents"]
            )
            
            v_metadatas = results["metadatas"][0] if results["metadatas"] else []
            v_documents = results["documents"][0] if results["documents"] else []
            
            candidates = []
            for doc, meta in zip(v_documents, v_metadatas):
                candidates.append({
                    "text": doc,
                    "metadata": meta,
                    "rrf_score": 1.0  # constant score since no fusion
                })
                
            if use_reranker and candidates:
                cross_encoder = retriever._get_cross_encoder()
                pairs = [(query_text, item["text"]) for item in candidates]
                scores = cross_encoder.predict(pairs)
                for item, score in zip(candidates, scores):
                    item["rerank_score"] = float(score)
                retrieved_chunks = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
            else:
                retrieved_chunks = candidates[:top_k]

        # Process metadata & sources
        sources = []
        contexts = []
        for item in retrieved_chunks:
            src = item["metadata"].get("source", "")
            if src and src not in sources:
                sources.append(src)
            contexts.append(item["text"])

        # Generation step
        answer = self.generator.generate_answer(query_text, contexts)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_sources": sources,
            "retrieved_chunks": [
                {
                    "text": item["text"],
                    "source": item["metadata"].get("source", ""),
                    "rrf_score": item.get("rrf_score", 0.0),
                    "rerank_score": item.get("rerank_score", 0.0)
                }
                for item in retrieved_chunks
            ]
        }

if __name__ == "__main__":
    # Test end-to-end unified pipeline
    pipeline = RAGPipeline()
    res = pipeline.query(
        "How do you serve static files in FastAPI?",
        strategy="semantic",
        use_hybrid=True,
        use_reranker=True
    )
    print("\n" + "="*50)
    print("QUERY:", res["query"])
    print("SOURCES:", res["retrieved_sources"])
    print("ANSWER:\n", res["answer"])
    print("="*50)
