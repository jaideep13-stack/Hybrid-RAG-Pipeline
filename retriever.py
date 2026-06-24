"""
retriever.py — Hybrid Retriever with Reciprocal Rank Fusion
============================================================
Hybrid retrieval = BM25 (sparse) + Dense vector search combined.

WHY HYBRID BEATS EITHER ALONE:
  BM25 alone misses:
    - Synonyms ("car" won't match "automobile")
    - Paraphrases ("help" won't match "assist")
    - Conceptual questions with no keyword overlap

  Dense alone misses:
    - Exact rare keywords (product codes, names, abbreviations)
    - Short precise queries ("CVE-2021-44228")
    - When training data didn't cover the domain

  Hybrid catches both. Empirically 5-15% better recall@k than either alone.

RECIPROCAL RANK FUSION (RRF):
  Simple, robust way to merge ranked lists from different systems.
  No need to normalize scores (BM25 and cosine scores are on different scales).

  RRF_score(doc) = sum over each ranker of: 1 / (k + rank_in_that_ranker)

  Where k=60 is a constant (empirically tuned, mostly insensitive to exact value).

  Key insight: uses RANK not RAW SCORE. Rank 1 always contributes 1/61,
  rank 2 contributes 1/62, etc. Doesn't matter if BM25 score is 14.3
  and cosine is 0.87 — they're incomparable anyway.

  A doc ranked #1 by BM25 and #3 by dense: RRF = 1/61 + 1/63 = 0.0322
  A doc ranked #5 by both:                 RRF = 1/65 + 1/65 = 0.0308
  → Both rankers agreeing on rank 5 beats one ranker alone at rank 1.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from bm25 import BM25
from vector_store import FlatVectorStore, SearchResult
from embedder import DenseEmbedder, TFIDFEmbedder
from chunker import Chunk


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: Dict
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    retrieval_method: str = "hybrid"


# ---------------------------------------------------------------------------
# RECIPROCAL RANK FUSION
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF.

    ranked_lists: list of ranked result lists, each is [(doc_id, score), ...]
                  sorted by score descending
    k:            RRF constant (default 60, from original paper)
    weights:      optional weight per ranker (default: equal weights)

    Returns: merged list of (doc_id, rrf_score) sorted descending
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    assert len(weights) == len(ranked_lists)

    rrf_scores: Dict[str, float] = {}

    for ranker_idx, ranked_list in enumerate(ranked_lists):
        w = weights[ranker_idx]
        for rank, (doc_id, _score) in enumerate(ranked_list):
            rrf_contribution = w / (k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution

    # Sort by RRF score descending
    merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return merged


# ---------------------------------------------------------------------------
# SPARSE RETRIEVER (BM25 only)
# ---------------------------------------------------------------------------

class SparseRetriever:
    """Wrapper around BM25 with the standard retriever interface."""

    def __init__(self):
        self.bm25 = BM25(k1=1.5, b=0.75)
        self._indexed_chunks: Dict[str, Chunk] = {}
        self._built = False

    def index(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        metadata = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "text": c.text, **c.metadata}
            for c in chunks
        ]
        self.bm25.build(texts, metadata)
        self._indexed_chunks = {c.chunk_id: c for c in chunks}
        self._built = True
        print(f"[SparseRetriever] Indexed {len(chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        results = self.bm25.search(query, top_k=top_k)
        output = []
        for rank, (doc_idx, score, meta) in enumerate(results):
            output.append(RetrievalResult(
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                text=meta["text"],
                score=score,
                rank=rank + 1,
                metadata=meta,
                bm25_rank=rank + 1,
                retrieval_method="sparse"
            ))
        return output


# ---------------------------------------------------------------------------
# DENSE RETRIEVER
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Wrapper around FlatVectorStore + DenseEmbedder."""

    def __init__(self, embedding_dim: int = 384, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = DenseEmbedder(model_name=model_name)
        self.store = FlatVectorStore(embedding_dim=self.embedder.embedding_dim)
        self._built = False

    def index(self, chunks: List[Chunk], batch_size: int = 64):
        print(f"[DenseRetriever] Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = self.embedder.encode(batch)
            all_embeddings.append(embs)
            if (i // batch_size) % 5 == 0:
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

        embeddings = np.vstack(all_embeddings)
        metadata = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "text": c.text, **c.metadata}
            for c in chunks
        ]
        self.store.add(embeddings, metadata)
        self._built = True
        print(f"[DenseRetriever] Indexed {len(chunks)} chunks. Dim={embeddings.shape[1]}")

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        query_emb = self.embedder.encode_one(query)
        results = self.store.search(query_emb, top_k=top_k)
        output = []
        for r in results:
            output.append(RetrievalResult(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                text=r.text,
                score=r.score,
                rank=r.rank,
                metadata=r.metadata,
                dense_rank=r.rank,
                retrieval_method="dense"
            ))
        return output


# ---------------------------------------------------------------------------
# HYBRID RETRIEVER
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 and dense retrieval via RRF.

    Index once, query fast.
    Weights let you tune: bm25_weight=0 → pure dense, dense_weight=0 → pure BM25.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60
    ):
        self.sparse = SparseRetriever()
        self.dense = DenseRetriever(model_name=model_name)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        # Keep a lookup by chunk_id for fast result assembly
        self._chunk_lookup: Dict[str, Dict] = {}

    def index(self, chunks: List[Chunk]):
        """Index all chunks into both BM25 and dense store."""
        print(f"[HybridRetriever] Indexing {len(chunks)} chunks...")
        self.sparse.index(chunks)
        self.dense.index(chunks)

        # Build lookup
        for c in chunks:
            self._chunk_lookup[c.chunk_id] = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                **c.metadata
            }
        print(f"[HybridRetriever] Ready. {len(chunks)} chunks indexed.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,    # Fetch more from each system, then fuse
        return_scores: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve top-K results using hybrid RRF.

        fetch_k: how many to fetch from each sub-retriever before fusion.
                 Higher fetch_k = better recall at fusion, slightly slower.
        """
        # Get results from both systems (fetch more than top_k for better fusion)
        sparse_results = self.sparse.retrieve(query, top_k=fetch_k)
        dense_results = self.dense.retrieve(query, top_k=fetch_k)

        # Build ranked lists for RRF: [(chunk_id, score), ...]
        sparse_ranked = [(r.chunk_id, r.score) for r in sparse_results]
        dense_ranked = [(r.chunk_id, r.score) for r in dense_results]

        # Build rank lookup for sparse and dense
        sparse_rank_map = {r.chunk_id: r.rank for r in sparse_results}
        dense_rank_map = {r.chunk_id: r.rank for r in dense_results}

        # Apply RRF
        fused = reciprocal_rank_fusion(
            [sparse_ranked, dense_ranked],
            k=self.rrf_k,
            weights=[self.bm25_weight, self.dense_weight]
        )

        # Build output results
        results = []
        for rank, (chunk_id, rrf_score) in enumerate(fused[:top_k]):
            meta = self._chunk_lookup.get(chunk_id, {})
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                text=meta.get("text", ""),
                score=rrf_score,
                rank=rank + 1,
                metadata=meta,
                bm25_rank=sparse_rank_map.get(chunk_id),
                dense_rank=dense_rank_map.get(chunk_id),
                retrieval_method="hybrid_rrf"
            ))

        return results

    def retrieve_sparse_only(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.sparse.retrieve(query, top_k=top_k)

    def retrieve_dense_only(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.dense.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from chunker import load_from_string, DocumentChunker

    CORPUS = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances language models with external knowledge.
    BM25 is a classic sparse retrieval algorithm used in search engines like Elasticsearch.
    Dense retrieval uses neural embeddings to find semantically similar documents.
    Hybrid search combines sparse BM25 with dense vector search for better recall.
    Reciprocal Rank Fusion merges results from multiple retrieval systems without score normalization.
    FAISS enables fast approximate nearest neighbor search in high-dimensional vector spaces.
    Chunking splits documents into smaller pieces that fit within embedding model context windows.
    Rerankers improve retrieval precision by scoring query-document pairs with cross-encoders.
    The transformer architecture revolutionized natural language processing with attention mechanisms.
    Vector databases store embeddings and support fast similarity search at scale.
    Fine-tuning adapts pretrained models to domain-specific tasks with labeled data.
    Cosine similarity computes the angle between two embedding vectors to measure semantic closeness.
    """ * 5

    # Build chunks
    doc = load_from_string(CORPUS, doc_id="test_corpus")
    chunker = DocumentChunker(strategy="recursive", chunk_size=200, overlap=40, min_chunk_size=50)
    chunks = chunker.chunk_document(doc)
    print(f"Created {len(chunks)} chunks\n")

    # Test RRF directly
    print("--- RRF Test ---")
    list_a = [("doc1", 10.0), ("doc2", 8.0), ("doc3", 5.0)]
    list_b = [("doc2", 0.95), ("doc1", 0.80), ("doc4", 0.75)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    print("Fused ranking:")
    for doc_id, score in fused[:5]:
        print(f"  {doc_id}: {score:.5f}")

    # Test sparse retriever
    print("\n--- Sparse Retriever ---")
    sparse = SparseRetriever()
    sparse.index(chunks)
    results = sparse.retrieve("how does BM25 retrieval work", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.text[:80]}...")

    # Test hybrid retriever
    print("\n--- Hybrid Retriever ---")
    hybrid = HybridRetriever(bm25_weight=1.0, dense_weight=1.0)
    hybrid.index(chunks)

    query = "combine sparse and dense search"
    print(f"\nQuery: '{query}'")

    sparse_only = hybrid.retrieve_sparse_only(query, top_k=3)
    dense_only = hybrid.retrieve_dense_only(query, top_k=3)
    hybrid_results = hybrid.retrieve(query, top_k=3, fetch_k=10)

    print("\nSparse only:")
    for r in sparse_only:
        print(f"  [{r.score:.4f}] {r.text[:70]}...")

    print("\nDense only:")
    for r in dense_only:
        print(f"  [{r.score:.4f}] {r.text[:70]}...")

    print("\nHybrid (RRF):")
    for r in hybrid_results:
        print(f"  [rrf={r.score:.5f}] bm25_rank={r.bm25_rank} dense_rank={r.dense_rank}")
        print(f"    '{r.text[:70]}...'")
