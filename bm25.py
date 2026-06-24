"""
bm25.py — BM25 Sparse Retrieval from Scratch
=============================================
BM25 (Best Match 25) is the gold standard for keyword-based retrieval.
Used by Elasticsearch, Solr, and most search engines under the hood.

WHY BM25 OVER TF-IDF?
  TF-IDF has two problems:
  1. Term saturation: TF grows linearly — a word appearing 100x in a doc
     gets 100x weight vs appearing once. That's not useful.
  2. No length normalization: a 10,000 word doc will dominate retrieval
     just because it contains more words.

  BM25 fixes both:
  1. Saturation: TF is capped via k1 parameter. After enough occurrences,
     adding more barely increases score. Asymptotes to (k1+1).
  2. Length norm: divides by document length (b parameter controls strength).

BM25 FORMULA:
  For query Q with terms q1..qn and document D:

  score(D, Q) = sum over qi of:
    IDF(qi) * (tf(qi, D) * (k1 + 1)) / (tf(qi, D) + k1 * (1 - b + b * |D|/avgdl))

  Where:
    tf(qi, D)  = term frequency of qi in D
    |D|        = length of D in words
    avgdl      = average document length in corpus
    k1         = term saturation parameter (typically 1.2-2.0)
    b          = length normalization parameter (typically 0.75)
    IDF(qi)    = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)

PARAMETERS:
  k1=1.5, b=0.75 are the standard defaults.
  k1=0: no term frequency effect (pure IDF)
  b=0:  no length normalization
  b=1:  full length normalization
"""

import math
import json
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

from embedder import tokenize


# ---------------------------------------------------------------------------
# BM25 INDEX
# ---------------------------------------------------------------------------

class BM25:
    """
    BM25 retrieval index.

    Build: O(N * avg_doc_len) — tokenize and index all documents
    Query: O(|query_terms| * df) — only touch docs containing query terms
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        k1: term frequency saturation. Higher = more weight to term freq.
            Range: [1.2, 2.0]. k1=1.5 is a good default.
        b:  length normalization. b=0 = no normalization, b=1 = full.
            0.75 is standard.
        epsilon: floor for IDF scores (prevents negative IDF for very common terms).
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Corpus statistics (learned at index time)
        self.n_docs: int = 0
        self.avgdl: float = 0.0

        # Inverted index: term → {doc_idx: term_freq}
        # This is the core data structure — same as what search engines use
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)

        # Document lengths (in tokens)
        self.doc_lengths: List[int] = []

        # IDF scores per term (computed after all docs are indexed)
        self.idf: Dict[str, float] = {}

        # Metadata per document
        self.doc_metadata: List[Dict] = []

        self._built = False

    def build(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index all documents.

        documents: list of raw text strings
        metadata:  optional list of dicts (chunk_id, doc_id, text, etc.)
        """
        self.n_docs = len(documents)
        if metadata is None:
            metadata = [{"doc_idx": i, "text": doc} for i, doc in enumerate(documents)]
        self.doc_metadata = metadata

        print(f"[BM25] Indexing {self.n_docs} documents...")

        # Step 1: Tokenize and build inverted index
        total_tokens = 0
        for doc_idx, doc_text in enumerate(documents):
            tokens = tokenize(doc_text, remove_stopwords=False, stem=True)
            # BM25 typically doesn't remove stopwords — they're downweighted by IDF
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            # Count term frequencies in this doc
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term][doc_idx] = count

        # Step 2: Compute average document length
        self.avgdl = total_tokens / self.n_docs if self.n_docs > 0 else 1.0

        # Step 3: Compute IDF for all terms
        # IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        # The +1 at the end ensures IDF >= 0 (Robertson-Sparck Jones variant)
        for term, postings in self.inverted_index.items():
            df = len(postings)  # Number of docs containing this term
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = max(idf, self.epsilon)  # Floor at epsilon

        self._built = True
        print(f"[BM25] Built index. Vocab: {len(self.inverted_index):,} terms, "
              f"avgdl: {self.avgdl:.1f} tokens")

    def _bm25_score(self, term: str, doc_idx: int) -> float:
        """
        Compute BM25 score contribution for one term in one document.

        BM25_term = IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*|D|/avgdl))
        """
        if term not in self.inverted_index:
            return 0.0
        if doc_idx not in self.inverted_index[term]:
            return 0.0

        tf = self.inverted_index[term][doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        idf = self.idf.get(term, 0.0)

        # Length-normalized TF
        length_norm = 1 - self.b + self.b * (doc_len / self.avgdl)
        tf_normalized = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

        return idf * tf_normalized

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        Search for top-K documents matching the query.

        Returns list of (doc_idx, score, metadata) sorted by score descending.

        Efficiency trick: only score documents that contain at least one query term.
        This is the "inverted" part of inverted index — we look up which docs
        each term appears in, rather than scanning all docs.
        """
        if not self._built:
            raise RuntimeError("Call build() first.")

        query_tokens = tokenize(query, remove_stopwords=False, stem=True)
        if not query_tokens:
            return []

        # Accumulate scores per document
        doc_scores: Dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue  # Term not in corpus at all

            # Only score docs that contain this term (inverted index lookup)
            for doc_idx in self.inverted_index[term]:
                doc_scores[doc_idx] += self._bm25_score(term, doc_idx)

        if not doc_scores:
            return []

        # Sort by score and return top-K
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for doc_idx, score in sorted_docs:
            results.append((doc_idx, score, self.doc_metadata[doc_idx]))

        return results

    def get_term_stats(self, term: str) -> Dict:
        """Debug utility: show stats for a specific term."""
        stem_term = tokenize(term, remove_stopwords=False, stem=True)
        if not stem_term:
            return {}
        t = stem_term[0]
        return {
            "term": t,
            "idf": self.idf.get(t, 0.0),
            "doc_frequency": len(self.inverted_index.get(t, {})),
            "total_docs": self.n_docs,
        }

    def save(self, path: str):
        """Persist the index to disk."""
        os.makedirs(path, exist_ok=True)
        # Convert defaultdict to regular dict for JSON serialization
        index_data = {
            term: dict(postings)
            for term, postings in self.inverted_index.items()
        }
        with open(os.path.join(path, "inverted_index.json"), "w") as f:
            json.dump(index_data, f)
        with open(os.path.join(path, "idf.json"), "w") as f:
            json.dump(self.idf, f)
        with open(os.path.join(path, "doc_metadata.json"), "w") as f:
            json.dump(self.doc_metadata, f)
        config = {
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        print(f"[BM25] Saved to {path}")

    def load(self, path: str) -> "BM25":
        """Load index from disk."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        self.n_docs = config["n_docs"]
        self.avgdl = config["avgdl"]
        self.doc_lengths = config["doc_lengths"]
        self.k1 = config["k1"]
        self.b = config["b"]
        self.epsilon = config["epsilon"]

        with open(os.path.join(path, "inverted_index.json")) as f:
            raw = json.load(f)
            self.inverted_index = {
                term: {int(k): v for k, v in postings.items()}
                for term, postings in raw.items()
            }
        with open(os.path.join(path, "idf.json")) as f:
            self.idf = json.load(f)
        with open(os.path.join(path, "doc_metadata.json")) as f:
            self.doc_metadata = json.load(f)

        self._built = True
        print(f"[BM25] Loaded from {path}. Docs: {self.n_docs}, Vocab: {len(self.inverted_index):,}")
        return self


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus = [
        "Retrieval augmented generation combines information retrieval with language models to generate grounded responses.",
        "BM25 is a probabilistic retrieval model that ranks documents based on term frequency and inverse document frequency.",
        "Dense embeddings capture semantic meaning allowing retrieval of semantically similar documents.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "Hybrid retrieval combines sparse BM25 scores with dense embedding scores using reciprocal rank fusion.",
        "Chunking splits long documents into smaller pieces suitable for embedding models with limited context windows.",
        "Rerankers use cross-encoder models to score query-document pairs more accurately than bi-encoder retrievers.",
        "Vector databases store and index embeddings for fast approximate nearest neighbor search.",
        "The transformer architecture uses multi-head attention to process sequences in parallel.",
        "Fine-tuning adapts a pretrained language model to a specific downstream task with labeled data.",
        "RAG systems retrieve relevant context at inference time rather than storing all knowledge in model weights.",
        "Cosine similarity measures the angle between two vectors regardless of their magnitude.",
    ]

    metadata = [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": t} for i, t in enumerate(corpus)]

    print("=" * 60)
    print("BM25 Test")
    print("=" * 60)

    bm25 = BM25(k1=1.5, b=0.75)
    bm25.build(corpus, metadata)

    queries = [
        "how does BM25 ranking work",
        "semantic search with dense vectors",
        "combine sparse and dense retrieval",
        "document chunking for embedding models",
    ]

    for query in queries:
        results = bm25.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for doc_idx, score, meta in results:
            print(f"  [{score:.4f}] {meta['text'][:70]}...")

    # Term stats
    print("\n--- Term Stats ---")
    for term in ["retrieval", "bm25", "dense", "embedding"]:
        stats = bm25.get_term_stats(term)
        print(f"  '{term}': IDF={stats.get('idf', 0):.3f}, df={stats.get('doc_frequency', 0)}/{bm25.n_docs}")

    # Save/load
    bm25.save("/tmp/bm25_index")
    bm25_loaded = BM25()
    bm25_loaded.load("/tmp/bm25_index")
    results2 = bm25_loaded.search("BM25 retrieval ranking", top_k=2)
    print(f"\n✓ Save/load works. Top result: [{results2[0][1]:.4f}] {results2[0][2]['text'][:60]}...")

    # Show BM25 vs TF-IDF difference
    print("\n--- BM25 saturation demo ---")
    # Short doc with many repetitions vs long doc with single mention
    docs_demo = [
        "retrieval retrieval retrieval retrieval retrieval",   # 5x term
        "retrieval augmented generation combines dense sparse hybrid methods for better results",  # 1x term, more content
    ]
    bm25_demo = BM25(k1=1.5, b=0.75)
    bm25_demo.build(docs_demo)
    res = bm25_demo.search("retrieval", top_k=2)
    print("Query: 'retrieval'")
    for idx, score, meta in res:
        print(f"  Doc {idx}: score={score:.4f} | '{docs_demo[idx][:60]}'")
    print("(BM25 penalizes spam repetition — saturated TF vs. length normalization)")
