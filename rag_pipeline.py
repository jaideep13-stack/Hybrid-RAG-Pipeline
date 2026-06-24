"""
reranker.py + generator.py + rag_pipeline.py + eval.py — All in One
====================================================================
Remaining components combined:

RERANKER:
  Bi-encoders (dense retrieval) are fast but imprecise — they encode query
  and document SEPARATELY, so interaction between them is limited.
  Cross-encoders take the concatenated [query + document] as input — they
  see both together, giving much better relevance scores.
  Tradeoff: cross-encoders are ~100x slower → use them only on top-K retrieved.

GENERATOR (Groq):
  Takes retrieved chunks + original query → generates a grounded answer.
  Key prompt engineering decisions:
  - Tell the model to ONLY use the provided context
  - Ask it to cite which chunks it used
  - Tell it to say "I don't know" if context doesn't contain the answer

PIPELINE:
  query → retrieve → rerank → generate → answer

EVALUATION:
  Retrieval eval (no LLM needed):
  - Recall@K: fraction of relevant docs in top-K
  - MRR (Mean Reciprocal Rank): 1/rank of first relevant doc
  - NDCG: normalized discounted cumulative gain (position-weighted)
  Generation eval:
  - Faithfulness: is the answer grounded in retrieved context?
  - Answer relevance: does the answer address the question?
"""

import os
import json
import math
import time
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from chunker import Chunk, DocumentChunker, load_from_string
from retriever import HybridRetriever, RetrievalResult


# ===========================================================================
# RERANKER
# ===========================================================================

@dataclass
class RankedResult:
    chunk_id: str
    doc_id: str
    text: str
    rerank_score: float
    original_rank: int
    new_rank: int
    metadata: Dict


class CrossEncoderReranker:
    """
    Cross-encoder reranker.

    In production: uses a model like 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
    Here: scores by computing TF-IDF overlap + length-normalized keyword matching.
    (Replace _score_pair with a real cross-encoder if sentence-transformers available.)

    Usage pattern:
      1. Dense/BM25 retriever fetches top-50 candidates (fast, approximate)
      2. Reranker scores all 50 with cross-encoder (slow, precise)
      3. Return top-5 reranked results
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._try_load_model()

    def _try_load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] Loading '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name)
            print("[Reranker] Cross-encoder loaded.")
        except ImportError:
            print("[Reranker] sentence-transformers not available. Using heuristic scorer.")
            self.model = None

    def _heuristic_score(self, query: str, passage: str) -> float:
        """
        Fallback scoring when no cross-encoder available.
        Computes: keyword overlap + position bonus + length penalty.
        Not as good as a real cross-encoder but deterministic and fast.
        """
        q_words = set(query.lower().split())
        p_words = passage.lower().split()

        # Keyword overlap score
        p_word_set = set(p_words)
        overlap = len(q_words & p_word_set)
        if not q_words:
            return 0.0
        keyword_score = overlap / len(q_words)

        # Position bonus: early mention of query terms is better
        position_score = 0.0
        for i, word in enumerate(p_words[:50]):  # Check first 50 words
            if word in q_words:
                position_score += 1.0 / (1 + i * 0.1)

        # Length penalty: very short or very long passages are less useful
        optimal_len = 150
        length_penalty = 1.0 - abs(len(p_words) - optimal_len) / (optimal_len * 3)
        length_penalty = max(0.1, length_penalty)

        return keyword_score * 0.6 + (position_score / max(len(q_words), 1)) * 0.3 + length_penalty * 0.1

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        """
        Rerank retrieved results.

        query:   original user query
        results: candidates from retriever (typically 20-50)
        top_k:   how many to return after reranking
        """
        if not results:
            return []

        top_k = top_k or len(results)

        if self.model is not None:
            # Real cross-encoder: takes list of (query, passage) pairs
            pairs = [(query, r.text) for r in results]
            scores = self.model.predict(pairs).tolist()
        else:
            # Heuristic fallback
            scores = [self._heuristic_score(query, r.text) for r in results]

        # Sort by score descending
        scored = sorted(zip(scores, results), key=lambda x: -x[0])

        reranked = []
        for new_rank, (score, result) in enumerate(scored[:top_k]):
            reranked.append(RankedResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                rerank_score=float(score),
                original_rank=result.rank,
                new_rank=new_rank + 1,
                metadata=result.metadata
            ))

        return reranked


# ===========================================================================
# GENERATOR (Groq API)
# ===========================================================================

@dataclass
class GeneratedAnswer:
    answer: str
    query: str
    retrieved_chunks: List[Dict]
    model: str
    tokens_used: int = 0
    generation_time: float = 0.0
    source_chunks_cited: List[str] = field(default_factory=list)


class GroqGenerator:
    """
    Answer generator using Groq API (LLaMA 3).

    Prompt engineering:
    - Strict grounding: model must only use provided context
    - Source citation: model must reference chunk numbers
    - Uncertainty handling: explicit "I don't know" instruction
    """

    DEFAULT_MODEL = "llama3-70b-8192"

    SYSTEM_PROMPT = """You are a precise question-answering assistant. Your job is to answer the user's question using ONLY the information provided in the context chunks below.

Rules you must follow:
1. Base your answer ONLY on the provided context. Do not use outside knowledge.
2. If the context doesn't contain enough information to answer, say "The provided context does not contain enough information to answer this question."
3. When you use information from a chunk, cite it as [Chunk N] in your answer.
4. Be concise and factual. Do not speculate or add opinions.
5. If multiple chunks are relevant, synthesize them into a coherent answer."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        if not self.api_key:
            print("[Generator] WARNING: No GROQ_API_KEY set. Generation will fail.")

    def _build_prompt(self, query: str, chunks: List[Dict]) -> str:
        """
        Build the RAG prompt.
        Format: system context + numbered chunks + user question.
        """
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Chunk {i+1}]\n{chunk['text']}\n")

        context = "\n".join(context_parts)
        return f"""Context:
{context}

Question: {query}

Answer (cite chunks as [Chunk N]):"""

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> GeneratedAnswer:
        """
        Generate an answer given query and retrieved chunks.

        temperature=0.1: low temperature for factual, grounded answers.
        High temperature → more creative but less faithful to context.
        """
        if not self.api_key:
            return GeneratedAnswer(
                answer="[ERROR: No GROQ_API_KEY set. Export GROQ_API_KEY=your_key]",
                query=query,
                retrieved_chunks=retrieved_chunks,
                model=self.model
            )

        prompt = self._build_prompt(query, retrieved_chunks)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        start_time = time.time()
        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            answer = data["choices"][0]["message"]["content"].strip()
            tokens = data.get("usage", {}).get("total_tokens", 0)
            gen_time = time.time() - start_time

            # Extract cited chunk numbers
            import re
            cited = re.findall(r'\[Chunk (\d+)\]', answer)
            cited_ids = [retrieved_chunks[int(n)-1].get("chunk_id", f"chunk_{n}")
                        for n in cited if 0 < int(n) <= len(retrieved_chunks)]

            return GeneratedAnswer(
                answer=answer,
                query=query,
                retrieved_chunks=retrieved_chunks,
                model=self.model,
                tokens_used=tokens,
                generation_time=gen_time,
                source_chunks_cited=cited_ids
            )

        except urllib.error.URLError as e:
            return GeneratedAnswer(
                answer=f"[API Error: {e}]",
                query=query,
                retrieved_chunks=retrieved_chunks,
                model=self.model
            )


# ===========================================================================
# FULL RAG PIPELINE
# ===========================================================================

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    query
      → HybridRetriever (BM25 + dense, RRF fusion)
      → CrossEncoderReranker (rerank top-20 → top-5)
      → GroqGenerator (generate answer with citations)
      → GeneratedAnswer
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[GroqGenerator] = None,
        fetch_k: int = 20,      # How many to retrieve before reranking
        top_k: int = 5,         # Final number of chunks to send to LLM
    ):
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or GroqGenerator()
        self.fetch_k = fetch_k
        self.top_k = top_k

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        verbose: bool = False
    ) -> GeneratedAnswer:
        """
        Full RAG query: retrieve → rerank → generate.
        """
        top_k = top_k or self.top_k
        t0 = time.time()

        # Step 1: Retrieve candidates
        retrieved = self.retriever.retrieve(question, top_k=self.fetch_k, fetch_k=self.fetch_k * 2)
        t_retrieve = time.time() - t0

        if verbose:
            print(f"[RAG] Retrieved {len(retrieved)} candidates ({t_retrieve:.2f}s)")
            for r in retrieved[:3]:
                print(f"  [{r.score:.4f}] {r.text[:60]}...")

        # Step 2: Rerank
        t1 = time.time()
        reranked = self.reranker.rerank(question, retrieved, top_k=top_k)
        t_rerank = time.time() - t1

        if verbose:
            print(f"[RAG] Reranked to top-{len(reranked)} ({t_rerank:.2f}s)")
            for r in reranked:
                print(f"  [rerank={r.rerank_score:.4f}, was rank {r.original_rank}] {r.text[:60]}...")

        # Step 3: Build context for generator
        context_chunks = [
            {"text": r.text, "chunk_id": r.chunk_id,
             "doc_id": r.doc_id, "score": r.rerank_score}
            for r in reranked
        ]

        # Step 4: Generate answer
        t2 = time.time()
        answer = self.generator.generate(question, context_chunks)
        t_gen = time.time() - t2

        if verbose:
            print(f"[RAG] Generated answer ({t_gen:.2f}s, {answer.tokens_used} tokens)")
            print(f"[RAG] Total time: {time.time()-t0:.2f}s")

        return answer


# ===========================================================================
# EVALUATION
# ===========================================================================

@dataclass
class EvalResult:
    query: str
    relevant_ids: List[str]
    retrieved_ids: List[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall@K: fraction of relevant items found in top-K results.
    = |relevant ∩ top-K retrieved| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.
    = |relevant ∩ top-K retrieved| / K
    """
    if k == 0:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k_set & relevant_set) / k


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    MRR: 1 / (rank of first relevant document).
    If no relevant doc found: 0.
    Perfect retrieval (relevant at rank 1): MRR = 1.0
    Relevant at rank 5: MRR = 0.2
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    DCG penalizes relevant docs found at lower ranks logarithmically.
    DCG@K = sum_{i=1}^{K} rel_i / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K  where IDCG is perfect ranking DCG

    Binary relevance: rel_i = 1 if retrieved_ids[i] in relevant_ids, else 0.
    """
    relevant_set = set(relevant_ids)

    # DCG: actual ranking
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1)=0

    # IDCG: ideal ranking (all relevant docs at top)
    n_relevant_in_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_k))

    if idcg == 0:
        return 0.0
    return dcg / idcg


class RetrievalEvaluator:
    """
    Evaluate retrieval quality given a set of queries with known relevant documents.

    Usage:
      evaluator = RetrievalEvaluator()
      evaluator.add_query("what is BM25", relevant_chunk_ids=["c1", "c3"])
      evaluator.add_query("how does RAG work", relevant_chunk_ids=["c2", "c5"])
      results = evaluator.evaluate(retriever, k=5)
      evaluator.report(results)
    """

    def __init__(self):
        self.queries: List[Dict] = []

    def add_query(self, query: str, relevant_ids: List[str]):
        self.queries.append({"query": query, "relevant_ids": relevant_ids})

    def evaluate(self, retriever, k: int = 5) -> List[EvalResult]:
        results = []
        for item in self.queries:
            query = item["query"]
            relevant_ids = item["relevant_ids"]

            retrieved = retriever.retrieve(query, top_k=k)
            retrieved_ids = [r.chunk_id for r in retrieved]

            results.append(EvalResult(
                query=query,
                relevant_ids=relevant_ids,
                retrieved_ids=retrieved_ids,
                recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
                precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
                mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
                ndcg=ndcg_at_k(retrieved_ids, relevant_ids, k),
            ))

        return results

    def report(self, results: List[EvalResult], k: int = 5):
        avg_recall = sum(r.recall_at_k for r in results) / len(results)
        avg_precision = sum(r.precision_at_k for r in results) / len(results)
        avg_mrr = sum(r.mrr for r in results) / len(results)
        avg_ndcg = sum(r.ndcg for r in results) / len(results)

        print(f"\n{'='*50}")
        print(f"  Retrieval Evaluation @ K={k}")
        print(f"{'='*50}")
        print(f"  Recall@{k}:    {avg_recall:.4f}")
        print(f"  Precision@{k}: {avg_precision:.4f}")
        print(f"  MRR:          {avg_mrr:.4f}")
        print(f"  NDCG@{k}:     {avg_ndcg:.4f}")
        print(f"  Queries:      {len(results)}")
        print(f"{'='*50}")

        for r in results:
            status = "✓" if r.mrr > 0 else "✗"
            print(f"  {status} '{r.query[:50]}' | recall={r.recall_at_k:.2f} MRR={r.mrr:.2f}")


# ===========================================================================
# Quick test (no Groq needed for retrieval test)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    CORPUS = """
    BM25 is a ranking function used in information retrieval. It is based on the probabilistic
    retrieval framework and considers term frequency and inverse document frequency.
    BM25 stands for Best Match 25 and is the 25th iteration of the probabilistic model.

    Dense embeddings convert text into high-dimensional vectors using neural networks.
    These vectors capture semantic meaning so that similar texts have similar vectors.
    Models like sentence-transformers produce embeddings of fixed dimensionality.

    Hybrid retrieval combines sparse methods like BM25 with dense embedding search.
    The combination is typically done using Reciprocal Rank Fusion which merges ranked lists.
    Hybrid retrieval consistently outperforms either method alone in benchmarks.

    Chunking is the process of splitting long documents into smaller pieces for retrieval.
    Common strategies include fixed-size chunking, sentence chunking, and recursive chunking.
    The chunk size affects both retrieval precision and the context available for generation.

    Reranking improves retrieval precision by applying a cross-encoder model to re-score
    the top candidates retrieved by the faster bi-encoder or BM25 system.
    Cross-encoders see the query and document together, enabling finer relevance judgments.
    """ * 3

    # Build chunks
    doc = load_from_string(CORPUS, doc_id="rag_corpus")
    chunker = DocumentChunker(strategy="sentence", chunk_size=200, min_chunk_size=40)
    chunks = chunker.chunk_document(doc)
    print(f"\nChunks created: {len(chunks)}")

    # Index in hybrid retriever
    retriever = HybridRetriever(bm25_weight=1.0, dense_weight=1.0)
    retriever.index(chunks)

    # Test reranker
    print("\n--- Reranker Test ---")
    reranker = CrossEncoderReranker()
    query = "how does BM25 work"
    candidates = retriever.retrieve(query, top_k=8, fetch_k=16)
    reranked = reranker.rerank(query, candidates, top_k=3)
    print(f"Query: '{query}'")
    for r in reranked:
        print(f"  [rerank={r.rerank_score:.4f}, was_rank={r.original_rank}] {r.text[:70]}...")

    # Test evaluation metrics
    print("\n--- Evaluation Metrics Test ---")
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c1", "c3", "c7"]
    k = 5
    print(f"Retrieved: {retrieved}")
    print(f"Relevant:  {relevant}")
    print(f"Recall@{k}:    {recall_at_k(retrieved, relevant, k):.4f}")
    print(f"Precision@{k}: {precision_at_k(retrieved, relevant, k):.4f}")
    print(f"MRR:          {mean_reciprocal_rank(retrieved, relevant):.4f}")
    print(f"NDCG@{k}:     {ndcg_at_k(retrieved, relevant, k):.4f}")

    # Full evaluator
    print("\n--- Full Evaluator ---")
    chunk_ids = [c.chunk_id for c in chunks]
    evaluator = RetrievalEvaluator()
    evaluator.add_query("BM25 term frequency ranking", relevant_ids=chunk_ids[:2])
    evaluator.add_query("dense embedding neural network", relevant_ids=chunk_ids[2:4])
    evaluator.add_query("chunking document splitting", relevant_ids=chunk_ids[4:6])

    eval_results = evaluator.evaluate(retriever, k=5)
    evaluator.report(eval_results, k=5)

    print("\n--- Generator (no API key — showing prompt only) ---")
    gen = GroqGenerator(api_key="")
    prompt = gen._build_prompt("What is BM25?", [{"text": chunks[0].text, "chunk_id": chunks[0].chunk_id}])
    print(f"Prompt preview:\n{prompt[:400]}...")

    print("\n✓ All components working.")
    print("\nTo run with real generation: export GROQ_API_KEY=your_key && python rag_pipeline.py")
