# RAG Retrieval Benchmark Report

## 1. Dataset and Evaluation Methodology

- **Domain:** FastAPI documentation
- **Dataset:** `evaluation/evaluation.json`
- **Number of evaluation questions:** 50
- **Embedding model:** `all-MiniLM-L6-v2`
- **Retrieval cutoff:** Top 5
- **Ground truth:** Each question contains one or more expected source document paths.

A question counts as a hit at K if at least one expected source appears among the top K retrieved results.

### Metrics

- **Hit@1:** Fraction of questions with a correct source at rank 1.
- **Hit@3:** Fraction of questions with a correct source in the top 3.
- **Hit@5:** Fraction of questions with a correct source in the top 5.

## 2. Chunking-Only Retrieval Results

Source: `evaluation/retrieval_results.json`

| Strategy | Hit@1 | Hit@3 | Hit@5 | Questions |
|---|---:|---:|---:|---:|
| Fixed | 54% (27/50) | 84% (42/50) | 90% (45/50) | 50 |
| Recursive | 44% (22/50) | 82% (41/50) | 90% (45/50) | 50 |
| Semantic | 66% (33/50) | 86% (43/50) | 90% (45/50) | 50 |

## 3. Unified Retrieval Benchmark Results

Sources: `evaluation/benchmark_results.json` and `evaluation/benchmark_report.csv`

| Strategy | Hit@1 | Hit@3 | Hit@5 | Questions |
|---|---:|---:|---:|---:|
| Fixed | 54% | 84% | 90% | 50 |
| Semantic | 66% | 86% | 90% | 50 |
| Hybrid (BM25 + vector search with RRF) | 68% | 92% | 96% | 50 |
| Hybrid + cross-encoder reranker | 80% | 94% | 96% | 50 |

The unified benchmark currently reports fixed, semantic, hybrid, and hybrid-plus-reranking strategies. It does not report a recursive-only result.

## 4. Observations

1. Semantic retrieval has the highest Hit@1 among the three chunking-only baselines in Section 2.
2. Hybrid retrieval has a stored Hit@5 of 96%, compared with 90% for fixed and semantic retrieval in Section 3.
3. Adding the reranker is associated with a higher stored Hit@1 in the unified benchmark: 80% compared with 68% for hybrid retrieval without reranking.
4. These are retrieval metrics. They do not establish end-to-end answer correctness, faithfulness, latency, or operating cost.

## 5. Reproducibility and CI

The evaluation dataset is stored in `evaluation/evaluation.json`.

The retrieval evaluation script is `src/evaluation/run_retrieval_evaluation.py`. The unified benchmark script is `src/evaluation/run_full_benchmark.py`.

The current GitHub Actions workflow runs the chunking-only retrieval evaluation and its quality gate. It does not run the unified benchmark script or automatically publish its generated JSON and CSV as workflow artifacts.

Therefore, the committed result files should be treated as stored benchmark snapshots until the corresponding evaluation is rerun and verified.

To regenerate the unified benchmark locally, run:

```bash
python src/evaluation/run_full_benchmark.py
```

Review the generated JSON and CSV together before committing updated results.

## 6. Historical Results

Earlier 10-question and 20-question experiments used different benchmark sizes and reported different scores. They are historical experiments and must not be presented as current results for the 50-question dataset.

