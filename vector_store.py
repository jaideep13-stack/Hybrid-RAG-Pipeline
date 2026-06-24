"""
vector_store.py — Vector Store with Cosine Similarity Search from Scratch
=========================================================================
A vector store does two things:
  1. Store document embeddings (index them)
  2. Given a query embedding, find the K most similar stored embeddings fast

WHAT WE BUILD:
  - Flat (exact) search: brute-force cosine similarity against all vectors
  - IVF (Inverted File Index): cluster vectors into buckets, only search
    nearby buckets at query time — trades tiny accuracy loss for big speed gain
  - Persistence: save/load index to disk

WHY NOT JUST USE FAISS?
  Building it yourself means you understand:
  - Why cosine sim = dot product for normalized vectors
  - What IVF actually does (k-means clusters as "buckets")
  - Why approximate search exists (exact search is O(N*d), too slow at scale)

COMPLEXITY:
  Flat search:  O(N * d) per query  — exact, slow at N > 100k
  IVF search:   O(sqrt(N) * d) approx — fast, slight recall drop
"""

import numpy as np
import json
import os
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Dict
    rank: int = 0


# ---------------------------------------------------------------------------
# FLAT (EXACT) VECTOR STORE
# ---------------------------------------------------------------------------

class FlatVectorStore:
    """
    Brute-force exact cosine similarity search.

    Storage:
      - embeddings matrix: [N, d] float32
      - metadata list: [{chunk_id, doc_id, text, ...}]

    Search: compute dot product of query against all rows,
            return top-K indices. O(N*d) per query.

    Best for: N < 50,000 documents. Simple, exact, no approximation error.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.embeddings: Optional[np.ndarray] = None   # [N, d]
        self.metadata: List[Dict] = []
        self.n_vectors = 0

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add vectors to the index.

        embeddings: [n, d] float32 — must be L2-normalized
        metadata:   list of dicts with chunk_id, doc_id, text, etc.
        """
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Expected dim {self.embedding_dim}, got {embeddings.shape[1]}"
        assert len(embeddings) == len(metadata), \
            "embeddings and metadata must have same length"

        # Normalize if not already (defensive)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

        self.metadata.extend(metadata)
        self.n_vectors = len(self.metadata)
        print(f"[FlatStore] Added {len(embeddings)} vectors. Total: {self.n_vectors}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Find top-K most similar vectors to query.

        Since all vectors are L2-normalized:
        cosine_similarity(q, d) = q · d  (dot product)

        So we just do one matrix-vector multiply: scores = embeddings @ query
        Then pick top-K indices.
        """
        if self.embeddings is None or self.n_vectors == 0:
            return []

        # Normalize query
        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Cosine similarity = dot product (for normalized vectors)
        # scores: [N] — one score per stored vector
        scores = self.embeddings @ q   # Matrix multiply: [N, d] @ [d] = [N]

        # Get top-K indices (argpartition is faster than full argsort for large N)
        if top_k >= self.n_vectors:
            top_indices = np.arange(self.n_vectors)
        else:
            # argpartition puts top-k elements at the end (unsorted)
            top_indices = np.argpartition(scores, -top_k)[-top_k:]

        # Sort the top-k by score (descending)
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices):
            meta = self.metadata[idx]
            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                doc_id=meta.get("doc_id", ""),
                score=float(scores[idx]),
                text=meta.get("text", ""),
                metadata=meta,
                rank=rank + 1
            ))

        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all vectors belonging to a document."""
        keep_mask = [m.get("doc_id") != doc_id for m in self.metadata]
        n_removed = sum(1 for k in keep_mask if not k)

        if n_removed == 0:
            return 0

        keep_indices = [i for i, k in enumerate(keep_mask) if k]
        self.metadata = [self.metadata[i] for i in keep_indices]
        self.embeddings = self.embeddings[keep_indices] if keep_indices else None
        self.n_vectors = len(self.metadata)

        print(f"[FlatStore] Removed {n_removed} vectors for doc_id='{doc_id}'")
        return n_removed

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"embedding_dim": self.embedding_dim, "n_vectors": self.n_vectors}, f)
        print(f"[FlatStore] Saved {self.n_vectors} vectors to {path}")

    def load(self, path: str) -> "FlatVectorStore":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        self.embedding_dim = config["embedding_dim"]
        self.n_vectors = config["n_vectors"]

        emb_path = os.path.join(path, "embeddings.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)

        with open(os.path.join(path, "metadata.json")) as f:
            self.metadata = json.load(f)

        print(f"[FlatStore] Loaded {self.n_vectors} vectors from {path}")
        return self


# ---------------------------------------------------------------------------
# IVF (INVERTED FILE INDEX) VECTOR STORE
# ---------------------------------------------------------------------------

class IVFVectorStore:
    """
    Approximate Nearest Neighbor search using IVF (Inverted File Index).

    HOW IT WORKS:
      Training phase:
        1. Run k-means on all vectors to find `n_clusters` centroids
        2. Assign each vector to its nearest centroid (cluster)
        3. Store vectors grouped by cluster

      Search phase:
        1. Find the `n_probe` nearest centroids to the query
        2. Search ONLY vectors in those clusters
        3. Return top-K from that subset

    WHY THIS IS FASTER:
      If N = 1,000,000 vectors and n_clusters = 1000:
        - Each cluster has ~1000 vectors
        - We search n_probe=10 clusters → 10,000 vectors
        - That's 100x faster than searching all 1,000,000
        - At the cost of potentially missing some true nearest neighbors

    RECALL vs SPEED TRADEOFF:
      n_probe=1:   fastest, lowest recall
      n_probe=n_clusters: same as flat search (exact), slowest
      n_probe=sqrt(n_clusters): sweet spot (typical default)
    """

    def __init__(self, embedding_dim: int, n_clusters: int = 100, n_probe: int = 10):
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.n_probe = min(n_probe, n_clusters)

        self.centroids: Optional[np.ndarray] = None   # [n_clusters, d]
        self.cluster_vectors: List[np.ndarray] = []   # List of [n_i, d] arrays
        self.cluster_metadata: List[List[Dict]] = []  # Metadata per cluster
        self.n_vectors = 0
        self._trained = False

    def _kmeans(self, vectors: np.ndarray, n_clusters: int, n_iter: int = 20) -> np.ndarray:
        """
        K-means clustering from scratch.

        Returns centroids: [n_clusters, d]
        """
        N, d = vectors.shape
        n_clusters = min(n_clusters, N)

        print(f"[IVF] Running k-means: {N} vectors → {n_clusters} clusters ({n_iter} iters)...")

        # Initialize centroids by random sampling (k-means++ would be better)
        idx = np.random.choice(N, size=n_clusters, replace=False)
        centroids = vectors[idx].copy()

        for iteration in range(n_iter):
            # Step 1: Assign each vector to nearest centroid
            # distances: [N, n_clusters] — use dot product since vectors are normalized
            similarities = vectors @ centroids.T   # [N, n_clusters]
            assignments = np.argmax(similarities, axis=1)   # [N]

            # Step 2: Update centroids as mean of assigned vectors
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(n_clusters)

            for i, cluster_id in enumerate(assignments):
                new_centroids[cluster_id] += vectors[i]
                counts[cluster_id] += 1

            # Handle empty clusters (reinitialize to random vectors)
            for c in range(n_clusters):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    new_centroids[c] = vectors[np.random.randint(N)]

            # Normalize centroids (since we use cosine similarity)
            norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
            new_centroids = new_centroids / np.maximum(norms, 1e-8)

            # Check convergence
            shift = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids

            if iteration % 5 == 0:
                print(f"  Iter {iteration}: centroid shift = {shift:.6f}")

            if shift < 1e-6:
                print(f"  Converged at iter {iteration}")
                break

        return centroids

    def train(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Train the IVF index: cluster vectors and assign to clusters.
        """
        N, d = embeddings.shape
        assert d == self.embedding_dim

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        # Train k-means
        self.centroids = self._kmeans(embeddings, self.n_clusters)

        # Assign vectors to clusters
        similarities = embeddings @ self.centroids.T   # [N, n_clusters]
        assignments = np.argmax(similarities, axis=1)

        # Initialize cluster storage
        self.cluster_vectors = [[] for _ in range(self.n_clusters)]
        self.cluster_metadata = [[] for _ in range(self.n_clusters)]

        for i, cluster_id in enumerate(assignments):
            self.cluster_vectors[cluster_id].append(embeddings[i])
            self.cluster_metadata[cluster_id].append(metadata[i])

        # Convert to numpy arrays
        self.cluster_vectors = [
            np.array(vecs) if vecs else np.zeros((0, d))
            for vecs in self.cluster_vectors
        ]

        self.n_vectors = N
        self._trained = True

        cluster_sizes = [len(m) for m in self.cluster_metadata]
        print(f"[IVF] Trained. Avg cluster size: {np.mean(cluster_sizes):.0f}, "
              f"Max: {max(cluster_sizes)}, Min: {min(cluster_sizes)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Approximate nearest neighbor search.
        1. Find n_probe nearest centroids
        2. Search only those clusters
        """
        if not self._trained:
            raise RuntimeError("Call train() first.")

        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Find n_probe nearest centroids
        centroid_scores = self.centroids @ q   # [n_clusters]
        probe_clusters = np.argsort(centroid_scores)[-self.n_probe:]

        # Search vectors in those clusters
        candidate_scores = []
        candidate_meta = []

        for cluster_id in probe_clusters:
            vecs = self.cluster_vectors[cluster_id]
            metas = self.cluster_metadata[cluster_id]

            if len(vecs) == 0:
                continue

            scores = vecs @ q   # [n_in_cluster]
            for score, meta in zip(scores, metas):
                candidate_scores.append(float(score))
                candidate_meta.append(meta)

        if not candidate_scores:
            return []

        # Sort all candidates and return top-K
        sorted_indices = np.argsort(candidate_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(sorted_indices):
            meta = candidate_meta[idx]
            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                doc_id=meta.get("doc_id", ""),
                score=candidate_scores[idx],
                text=meta.get("text", ""),
                metadata=meta,
                rank=rank + 1
            ))

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.centroids is not None:
            np.save(os.path.join(path, "centroids.npy"), self.centroids)
        with open(os.path.join(path, "cluster_metadata.pkl"), "wb") as f:
            pickle.dump(self.cluster_metadata, f)
        for i, vecs in enumerate(self.cluster_vectors):
            if len(vecs) > 0:
                np.save(os.path.join(path, f"cluster_{i}.npy"), vecs)
        config = {
            "embedding_dim": self.embedding_dim,
            "n_clusters": self.n_clusters,
            "n_probe": self.n_probe,
            "n_vectors": self.n_vectors
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        print(f"[IVF] Saved to {path}")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    DIM = 64
    N = 200

    print("=" * 60)
    print("Vector Store Tests")
    print("=" * 60)

    # Generate random normalized vectors
    vecs = np.random.randn(N, DIM).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    metadata = [
        {"chunk_id": f"chunk_{i}", "doc_id": f"doc_{i%10}",
         "text": f"Document chunk number {i} about topic {i%5}"}
        for i in range(N)
    ]

    # --- Flat store ---
    print("\n--- FlatVectorStore ---")
    flat = FlatVectorStore(embedding_dim=DIM)
    flat.add(vecs[:100], metadata[:100])
    flat.add(vecs[100:], metadata[100:])

    query = np.random.randn(DIM).astype(np.float32)
    query = query / np.linalg.norm(query)

    results = flat.search(query, top_k=5)
    print(f"Query results (top 5):")
    for r in results:
        print(f"  rank={r.rank} score={r.score:.4f} id={r.chunk_id}")

    # Verify scores are descending
    scores = [r.score for r in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores not sorted!"
    print("✓ Scores correctly sorted (descending)")

    # Delete test
    n_removed = flat.delete_by_doc_id("doc_0")
    print(f"✓ Removed {n_removed} vectors for doc_0")

    # Save/load
    flat.save("/tmp/flat_store")
    flat2 = FlatVectorStore(embedding_dim=DIM)
    flat2.load("/tmp/flat_store")
    results2 = flat2.search(query, top_k=3)
    print(f"✓ Save/load works. Top result after reload: score={results2[0].score:.4f}")

    # --- IVF store ---
    print("\n--- IVFVectorStore ---")
    ivf = IVFVectorStore(embedding_dim=DIM, n_clusters=10, n_probe=3)
    ivf.train(vecs, metadata)

    ivf_results = ivf.search(query, top_k=5)
    print(f"\nIVF results (top 5):")
    for r in ivf_results:
        print(f"  rank={r.rank} score={r.score:.4f} id={r.chunk_id}")

    # Compare flat vs IVF
    flat_ids = {r.chunk_id for r in flat.search(query, top_k=10)}
    ivf_ids = {r.chunk_id for r in ivf.search(query, top_k=10)}
    overlap = len(flat_ids & ivf_ids)
    print(f"\nFlat vs IVF top-10 overlap: {overlap}/10 (recall@10 = {overlap/10:.1f})")
    print("(Some miss is expected — IVF trades recall for speed)")
