"""
embedder.py — TF-IDF + Dense Embeddings from Scratch
=====================================================
Two embedding approaches for RAG:

1. TF-IDF EMBEDDINGS (sparse)
   Term Frequency × Inverse Document Frequency.
   Each document becomes a sparse vector over the entire vocabulary.
   - TF(t, d)  = count of term t in document d / total terms in d
   - IDF(t)    = log(N / df(t)) where N=total docs, df(t)=docs containing t
   - TF-IDF(t,d) = TF * IDF
   Good for exact keyword matching. Bad for synonyms/semantic similarity.

2. DENSE EMBEDDINGS (via Groq API or local sentence-transformers)
   Maps text to a dense low-dimensional vector (e.g., 384 or 768 dims).
   Similar meaning → similar vectors (high cosine similarity).
   Uses a pre-trained model (we call Groq's embedding endpoint).
   Good for semantic similarity. Bad for exact rare-keyword matching.

WHY BOTH?
   Hybrid RAG uses both:
   - Sparse (TF-IDF/BM25) catches exact keyword matches
   - Dense catches semantic matches ("automobile" ≈ "car")
   - Combined: better recall than either alone
"""

import math
import json
import re
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "as", "by", "from",
    "this", "that", "these", "those", "be", "are", "was", "were",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "can", "its", "their", "there", "then", "than", "so", "if",
    "about", "up", "out", "into", "through", "over", "after",
    "also", "which", "who", "what", "when", "where", "how", "all",
    "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "only", "own", "same", "too",
    "very", "just", "because", "while", "although", "however"
}


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    """
    Tokenize text into words.
    - Lowercase
    - Remove punctuation
    - Optional stopword removal
    - Optional stemming (simple suffix stripping)
    """
    # Lowercase and split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'\b[a-z][a-z0-9]*\b', text)

    # Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # Simple suffix stemming (Porter-lite)
    if stem:
        tokens = [simple_stem(t) for t in tokens]

    return tokens


def simple_stem(word: str) -> str:
    """
    Very simple suffix stripping (not full Porter stemmer).
    Handles common English suffixes to group word variants.
    e.g.: "retrieval" → "retriev", "retrieving" → "retriev"
    """
    suffixes = ["ing", "tion", "ness", "ment", "ity", "ies",
                "ed", "er", "est", "ly", "al", "ive", "ous"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


# ---------------------------------------------------------------------------
# TF-IDF EMBEDDER
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """
    TF-IDF vectorizer built from scratch.

    Training:
      1. Tokenize all documents
      2. Build vocabulary (mapping word → index)
      3. Compute IDF for each word

    Encoding:
      1. Tokenize query/document
      2. Compute TF for each token
      3. Multiply by stored IDF
      4. Normalize the resulting vector

    The result is a SPARSE vector (most entries are 0).
    We store it as a dict {word_index: tfidf_value} for efficiency.
    """

    def __init__(
        self,
        max_features: int = 10000,   # Maximum vocabulary size
        min_df: int = 2,             # Minimum document frequency (ignore rare terms)
        max_df_ratio: float = 0.95,  # Maximum doc frequency ratio (ignore common terms)
        sublinear_tf: bool = True,   # Use log(1 + tf) instead of raw tf
        remove_stopwords: bool = True,
        stem: bool = True
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.sublinear_tf = sublinear_tf
        self.remove_stopwords = remove_stopwords
        self.stem = stem

        # Learned after fit()
        self.vocabulary: Dict[str, int] = {}      # word → index
        self.idf: Dict[str, float] = {}           # word → idf score
        self.vocab_size: int = 0
        self.n_docs: int = 0
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        """
        Learn vocabulary and IDF from a corpus.
        texts: list of raw text strings (one per document)
        """
        self.n_docs = len(texts)
        print(f"[TF-IDF] Fitting on {self.n_docs} documents...")

        # Step 1: Tokenize all documents and count document frequencies
        doc_freq = defaultdict(int)   # word → number of documents containing it
        tokenized_docs = []

        for text in texts:
            tokens = tokenize(text, self.remove_stopwords, self.stem)
            tokenized_docs.append(tokens)
            # Each word counts once per document (regardless of frequency in doc)
            for word in set(tokens):
                doc_freq[word] += 1

        # Step 2: Filter vocabulary by min_df and max_df
        max_df = int(self.max_df_ratio * self.n_docs)
        valid_words = {
            word for word, df in doc_freq.items()
            if self.min_df <= df <= max_df
        }

        # Step 3: Rank by document frequency and take top max_features
        sorted_words = sorted(valid_words, key=lambda w: -doc_freq[w])
        selected_words = sorted_words[:self.max_features]

        # Step 4: Build vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(selected_words)}
        self.vocab_size = len(self.vocabulary)

        # Step 5: Compute IDF for each word in vocabulary
        # IDF(t) = log((1 + N) / (1 + df(t))) + 1  (smoothed IDF)
        # The +1 prevents zero IDF and handles words seen in all docs
        for word in self.vocabulary:
            df = doc_freq[word]
            self.idf[word] = math.log((1 + self.n_docs) / (1 + df)) + 1.0

        self._is_fitted = True
        print(f"[TF-IDF] Vocabulary size: {self.vocab_size:,} (from {len(doc_freq):,} unique terms)")
        return self

    def transform_one(self, text: str) -> Dict[int, float]:
        """
        Encode a single text to a sparse TF-IDF vector.
        Returns: {word_index: tfidf_weight}
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        tokens = tokenize(text, self.remove_stopwords, self.stem)
        if not tokens:
            return {}

        # Count term frequency
        term_counts = Counter(tokens)
        total_terms = len(tokens)

        # Build TF-IDF vector
        vector = {}
        for word, count in term_counts.items():
            if word not in self.vocabulary:
                continue  # Out-of-vocabulary word

            idx = self.vocabulary[word]

            # Term frequency
            if self.sublinear_tf:
                tf = 1.0 + math.log(count)  # Sublinear TF dampens high counts
            else:
                tf = count / total_terms

            # TF-IDF weight
            tfidf = tf * self.idf[word]
            vector[idx] = tfidf

        # L2 normalize the vector
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            vector = {k: v / norm for k, v in vector.items()}

        return vector

    def transform(self, texts: List[str]) -> List[Dict[int, float]]:
        """Encode a list of texts."""
        return [self.transform_one(text) for text in texts]

    def sparse_dot(self, vec_a: Dict[int, float], vec_b: Dict[int, float]) -> float:
        """
        Dot product between two sparse vectors.
        Only iterate over non-zero entries for efficiency.
        """
        # Iterate over smaller vector for speed
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(val * vec_b.get(idx, 0.0) for idx, val in vec_a.items())

    def cosine_similarity(self, vec_a: Dict[int, float], vec_b: Dict[int, float]) -> float:
        """
        Cosine similarity between two pre-normalized sparse vectors.
        Since vectors are L2-normalized in transform_one, this equals dot product.
        """
        return self.sparse_dot(vec_a, vec_b)

    def to_dense(self, sparse_vec: Dict[int, float]) -> np.ndarray:
        """Convert sparse vector to dense numpy array."""
        dense = np.zeros(self.vocab_size)
        for idx, val in sparse_vec.items():
            dense[idx] = val
        return dense

    def save(self, path: str):
        data = {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "sublinear_tf": self.sublinear_tf,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[TF-IDF] Saved to {path}")

    def load(self, path: str) -> "TFIDFEmbedder":
        with open(path) as f:
            data = json.load(f)
        self.vocabulary = {k: int(v) for k, v in data["vocabulary"].items()}
        self.idf = data["idf"]
        self.n_docs = data["n_docs"]
        self.vocab_size = data["vocab_size"]
        self._is_fitted = True
        print(f"[TF-IDF] Loaded from {path} (vocab={self.vocab_size:,})")
        return self


# ---------------------------------------------------------------------------
# DENSE EMBEDDER (via sentence-transformers or Groq)
# ---------------------------------------------------------------------------

class DenseEmbedder:
    """
    Dense embedding using a pre-trained model.

    Strategy:
      - If sentence-transformers is available: use it locally (free, private)
      - Otherwise: use Groq embedding API

    Both return fixed-size dense vectors (e.g., 384 or 768 dims).
    Vectors are L2-normalized so cosine similarity = dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_groq: bool = False):
        self.model_name = model_name
        self.use_groq = use_groq
        self.model = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        """Try to load sentence-transformers locally."""
        if self.use_groq:
            print("[DenseEmbedder] Using Groq API for embeddings.")
            self.embedding_dim = 1024  # Groq embedding dim
            return

        try:
            from sentence_transformers import SentenceTransformer
            print(f"[DenseEmbedder] Loading '{self.model_name}' locally...")
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            test_emb = self.model.encode(["test"])
            self.embedding_dim = test_emb.shape[1]
            print(f"[DenseEmbedder] Loaded. Embedding dim: {self.embedding_dim}")
        except ImportError:
            print("[DenseEmbedder] sentence-transformers not installed. Falling back to random embeddings for demo.")
            print("  Install: pip install sentence-transformers")
            self.model = None
            self.embedding_dim = 384  # MiniLM dimension

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode texts to dense vectors.
        Returns: [n_texts, embedding_dim] numpy array
        """
        if self.model is not None:
            # Use sentence-transformers
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100
            )
            return np.array(embeddings)

        elif self.use_groq:
            return self._encode_groq(texts, normalize)

        else:
            # DEMO FALLBACK: random normalized vectors
            # In production, replace with a real model
            print("[DenseEmbedder] WARNING: Using random embeddings (demo mode only)")
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)
            return embeddings

    def _encode_groq(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Call Groq embedding API."""
        import urllib.request
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError("Set GROQ_API_KEY environment variable for Groq embeddings.")

        embeddings = []
        # Groq API: send in batches
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = json.dumps({
                "model": "llama3-groq-70b-8192-tool-use-preview",
                "input": batch
            }).encode()

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/embeddings",
                data=payload,
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                for item in data["data"]:
                    embeddings.append(item["embedding"])

        embeddings = np.array(embeddings, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        return embeddings

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# HYBRID EMBEDDER: wraps both
# ---------------------------------------------------------------------------

class HybridEmbedder:
    """
    Combines TF-IDF (sparse) and Dense embeddings.
    Used by the hybrid retriever.
    """

    def __init__(self, dense_model: str = "all-MiniLM-L6-v2"):
        self.tfidf = TFIDFEmbedder(max_features=10000, min_df=1)
        self.dense = DenseEmbedder(model_name=dense_model)
        self._fitted = False

    def fit(self, texts: List[str]) -> "HybridEmbedder":
        """Fit TF-IDF on corpus. Dense model is pre-trained (no fitting needed)."""
        self.tfidf.fit(texts)
        self._fitted = True
        return self

    def encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        return self.tfidf.transform(texts)

    def encode_dense(self, texts: List[str]) -> np.ndarray:
        return self.dense.encode(texts)

    def encode_one_sparse(self, text: str) -> Dict[int, float]:
        return self.tfidf.transform_one(text)

    def encode_one_dense(self, text: str) -> np.ndarray:
        return self.dense.encode_one(text)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TF-IDF Embedder Test")
    print("=" * 60)

    corpus = [
        "Retrieval augmented generation combines retrieval with language models.",
        "Dense embeddings capture semantic similarity between documents.",
        "BM25 is a sparse retrieval algorithm based on term frequency.",
        "FAISS enables fast approximate nearest neighbor search in vector spaces.",
        "Hybrid retrieval combines sparse and dense methods for better recall.",
        "Chunking splits documents into smaller pieces for embedding.",
        "The transformer architecture uses attention mechanisms for sequence modeling.",
        "Cross-encoder rerankers score query-document pairs for improved precision.",
        "Reciprocal rank fusion merges results from multiple retrieval systems.",
        "Cosine similarity measures the angle between two embedding vectors.",
    ]

    # Fit TF-IDF
    embedder = TFIDFEmbedder(max_features=500, min_df=1, sublinear_tf=True)
    embedder.fit(corpus)

    # Encode query and documents
    query = "how does semantic search work with embeddings"
    q_vec = embedder.transform_one(query)
    doc_vecs = embedder.transform(corpus)

    # Rank documents by cosine similarity to query
    scores = [(i, embedder.cosine_similarity(q_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scores.sort(key=lambda x: -x[1])

    print(f"\nQuery: '{query}'")
    print("\nRanked results (TF-IDF):")
    for rank, (doc_idx, score) in enumerate(scores[:5]):
        print(f"  {rank+1}. [score={score:.4f}] {corpus[doc_idx][:70]}...")

    # Test tokenization
    print(f"\n--- Tokenization ---")
    test = "The retrieval system uses BM25 ranking for documents"
    tokens = tokenize(test, remove_stopwords=True, stem=True)
    print(f"Input:  '{test}'")
    print(f"Tokens: {tokens}")

    # Sparse vector inspection
    print(f"\n--- Sparse Vector ---")
    vec = embedder.transform_one("dense embedding retrieval")
    print(f"Non-zero dimensions: {len(vec)}")
    # Show top-5 by weight
    top5 = sorted(vec.items(), key=lambda x: -x[1])[:5]
    inv_vocab = {v: k for k, v in embedder.vocabulary.items()}
    print("Top-5 TF-IDF weights:")
    for idx, weight in top5:
        word = inv_vocab.get(idx, "?")
        print(f"  '{word}': {weight:.4f}")

    # Dense embedder test
    print(f"\n--- Dense Embedder Test ---")
    dense = DenseEmbedder()  # Will use demo mode if no sentence-transformers
    embs = dense.encode(["hello world", "goodbye world"])
    print(f"Dense embedding shape: {embs.shape}")
    print(f"Embedding dim: {dense.embedding_dim}")

    # Cosine similarity between similar vs dissimilar texts
    sim_similar = float(np.dot(embs[0], embs[1]))
    print(f"Cosine sim ('hello world' vs 'goodbye world'): {sim_similar:.4f}")
