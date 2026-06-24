"""
chunker.py — Document Loading + Smart Chunking Strategies
==========================================================
Chunking is the FIRST and most critical step in RAG.

WHY CHUNKING MATTERS:
  - Embedding models have a token limit (typically 512 tokens)
  - If chunks are too large: the embedding averages over too much text,
    retrieval becomes imprecise ("needle in a haystack" problem)
  - If chunks are too small: chunks lose context, answers are incomplete
  - The goal: chunks that are semantically self-contained

CHUNKING STRATEGIES IMPLEMENTED:

1. FIXED SIZE CHUNKING
   Split every N characters with overlap.
   Fast, simple, no linguistic awareness.
   Problem: splits mid-sentence, mid-word.

2. SENTENCE CHUNKING
   Split on sentence boundaries (.!?)
   Group sentences until chunk_size reached.
   Better semantic coherence than fixed-size.

3. RECURSIVE CHUNKING
   Try to split on paragraph → sentence → word → character.
   Respects document structure before falling back to character splits.
   Best general-purpose strategy (used by LangChain by default).

4. SEMANTIC CHUNKING
   Embed each sentence, group sentences whose embeddings are similar.
   Split when embedding similarity drops below threshold.
   Most semantically coherent — but slowest.

Each chunk carries metadata: source, chunk_id, start_char, strategy used.
"""

import re
import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Raw document before chunking."""
    doc_id: str
    text: str
    metadata: Dict  # title, source, author, date, etc.

    def __post_init__(self):
        if not self.doc_id:
            # Auto-generate ID from content hash
            self.doc_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A text chunk ready for embedding and indexing."""
    chunk_id: str        # Unique identifier
    doc_id: str          # Which document this came from
    text: str            # The actual text content
    start_char: int      # Character offset in original document
    end_char: int        # End character offset
    chunk_index: int     # Position of this chunk in its document
    strategy: str        # Which chunking strategy was used
    metadata: Dict       # Inherited from parent document + chunk-level info

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["word_count"] = self.word_count
        return d


# ---------------------------------------------------------------------------
# STRATEGY 1: FIXED SIZE CHUNKING
# ---------------------------------------------------------------------------

def fixed_size_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64
) -> List[Tuple[str, int, int]]:
    """
    Split text into fixed-size character chunks with overlap.

    overlap: number of characters to repeat between adjacent chunks.
    This ensures context isn't lost at chunk boundaries.

    Returns list of (chunk_text, start_char, end_char).
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append((chunk_text, start, end))

        # Move forward by (chunk_size - overlap)
        # If overlap >= chunk_size, we'd loop forever — guard against it
        step = max(1, chunk_size - overlap)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# STRATEGY 2: SENTENCE CHUNKING
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> List[Tuple[str, int]]:
    """
    Split text into sentences, returning (sentence, start_char) pairs.

    Uses regex to find sentence boundaries (.!?) while handling:
    - Abbreviations (Mr., Dr., U.S.A.)
    - Decimal numbers (3.14)
    - Ellipsis (...)
    """
    # Sentence boundary pattern:
    # Match .!? followed by whitespace + capital letter (or end of string)
    # Negative lookbehind for common abbreviations
    sentence_pattern = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z])'
    )

    sentences = []
    last_end = 0

    for match in sentence_pattern.finditer(text):
        sentence = text[last_end:match.start() + 1].strip()
        if sentence:
            sentences.append((sentence, last_end))
        last_end = match.end()

    # Add the last sentence
    remaining = text[last_end:].strip()
    if remaining:
        sentences.append((remaining, last_end))

    return sentences


def sentence_chunk(
    text: str,
    chunk_size: int = 512,
    overlap_sentences: int = 1
) -> List[Tuple[str, int, int]]:
    """
    Group sentences into chunks, targeting chunk_size characters.
    overlap_sentences: how many sentences to repeat at chunk boundaries.

    Algorithm:
      1. Split text into sentences
      2. Greedily add sentences until chunk_size is reached
      3. Start next chunk with last `overlap_sentences` sentences
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [(text, 0, len(text))] if text.strip() else []

    chunks = []
    current_sentences = []
    current_start = sentences[0][1] if sentences else 0

    for sent_text, sent_start in sentences:
        # Would adding this sentence exceed chunk_size?
        current_text = " ".join(s for s, _ in current_sentences)
        if current_sentences and len(current_text) + len(sent_text) > chunk_size:
            # Save current chunk
            chunk_text = " ".join(s for s, _ in current_sentences)
            chunk_end = sent_start
            chunks.append((chunk_text.strip(), current_start, chunk_end))

            # Start next chunk with overlap
            overlap_sents = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            current_sentences = overlap_sents + [(sent_text, sent_start)]
            current_start = overlap_sents[0][1] if overlap_sents else sent_start
        else:
            current_sentences.append((sent_text, sent_start))

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(s for s, _ in current_sentences)
        chunks.append((chunk_text.strip(), current_start, len(text)))

    return chunks


# ---------------------------------------------------------------------------
# STRATEGY 3: RECURSIVE CHUNKING
# ---------------------------------------------------------------------------

def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    separators: Optional[List[str]] = None
) -> List[Tuple[str, int, int]]:
    """
    Recursively split on increasingly granular separators.

    Priority order:
      1. Double newline (paragraph break)
      2. Single newline
      3. Period + space (sentence)
      4. Comma + space
      5. Space (word)
      6. Character (last resort)

    If a split produces chunks still larger than chunk_size,
    recurse with the next separator.

    This is the most robust general-purpose strategy.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    def _split_recursive(text: str, separators: List[str], start_offset: int) -> List[Tuple[str, int, int]]:
        # Base case: text fits in one chunk
        if len(text) <= chunk_size:
            return [(text, start_offset, start_offset + len(text))] if text.strip() else []

        # No more separators — split by character
        if not separators:
            return fixed_size_chunk(text, chunk_size, overlap)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Character-level split
            return fixed_size_chunk(text, chunk_size, overlap)

        # Split by current separator
        parts = text.split(sep)

        if len(parts) == 1:
            # Separator not found — try next
            return _split_recursive(text, remaining_seps, start_offset)

        result_chunks = []
        current_chunk = ""
        current_start = start_offset
        char_pos = start_offset

        for i, part in enumerate(parts):
            # Reconstruct with separator (except last part)
            part_with_sep = part + (sep if i < len(parts) - 1 else "")

            if len(current_chunk) + len(part_with_sep) <= chunk_size:
                if not current_chunk:
                    current_start = char_pos
                current_chunk += part_with_sep
            else:
                # Save current chunk if non-empty
                if current_chunk.strip():
                    # If current_chunk itself is too large, recurse
                    if len(current_chunk) > chunk_size:
                        sub_chunks = _split_recursive(current_chunk, remaining_seps, current_start)
                        result_chunks.extend(sub_chunks)
                    else:
                        result_chunks.append((current_chunk.strip(), current_start, current_start + len(current_chunk)))

                    # Overlap: carry last `overlap` chars into next chunk
                    if overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + part_with_sep
                        current_start = current_start + len(current_chunk) - len(overlap_text) - len(part_with_sep)
                    else:
                        current_chunk = part_with_sep
                        current_start = char_pos
                else:
                    current_chunk = part_with_sep
                    current_start = char_pos

            char_pos += len(part_with_sep)

        # Last chunk
        if current_chunk.strip():
            result_chunks.append((current_chunk.strip(), current_start, char_pos))

        return result_chunks

    return _split_recursive(text, separators, 0)


# ---------------------------------------------------------------------------
# STRATEGY 4: SEMANTIC CHUNKING (embedding-based)
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tfidf_sentence_embedding(sentence: str, vocab: Dict[str, int]) -> List[float]:
    """
    Simple bag-of-words embedding for semantic chunking.
    Returns a sparse vector over vocab.
    (Real implementation uses a proper embedding model.)
    """
    words = sentence.lower().split()
    vec = [0.0] * len(vocab)
    for word in words:
        if word in vocab:
            vec[vocab[word]] += 1.0
    # L2 normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def semantic_chunk(
    text: str,
    chunk_size: int = 512,
    similarity_threshold: float = 0.5,
    embed_fn=None
) -> List[Tuple[str, int, int]]:
    """
    Split text at points where semantic similarity between adjacent
    sentences drops below similarity_threshold.

    embed_fn: function(sentence: str) -> List[float]
              If None, uses simple bag-of-words.

    Algorithm:
      1. Split into sentences
      2. Embed each sentence
      3. Compute cosine similarity between consecutive sentences
      4. Split where similarity < threshold (semantic boundary)
      5. Merge small groups to respect chunk_size
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return [(text, 0, len(text))]

    # Build simple vocabulary if no embed_fn provided
    if embed_fn is None:
        all_words = set()
        for sent, _ in sentences:
            all_words.update(sent.lower().split())
        vocab = {w: i for i, w in enumerate(sorted(all_words))}
        embed_fn = lambda s: tfidf_sentence_embedding(s, vocab)

    # Embed all sentences
    embeddings = [embed_fn(sent) for sent, _ in sentences]

    # Find semantic split points
    split_points = set()
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < similarity_threshold:
            split_points.add(i + 1)  # Split before sentence i+1

    # Build chunks from split points
    chunks = []
    current_group = []
    current_start = sentences[0][1]

    for i, (sent, sent_start) in enumerate(sentences):
        if i in split_points and current_group:
            # Check if current group is within chunk_size
            chunk_text = " ".join(current_group)
            if len(chunk_text) <= chunk_size:
                chunks.append((chunk_text.strip(), current_start, sent_start))
                current_group = [sent]
                current_start = sent_start
            else:
                # Too large — fall back to sentence chunking
                sub = sentence_chunk(chunk_text, chunk_size)
                offset = current_start
                for st, s, e in sub:
                    chunks.append((st, offset + s, offset + e))
                current_group = [sent]
                current_start = sent_start
        else:
            current_group.append(sent)

    # Last group
    if current_group:
        chunk_text = " ".join(current_group)
        chunks.append((chunk_text.strip(), current_start, len(text)))

    return chunks


# ---------------------------------------------------------------------------
# CHUNKER CLASS
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Main chunker class. Takes Documents, produces Chunks.

    Supports all four strategies with a unified interface.
    """

    STRATEGIES = ["fixed", "sentence", "recursive", "semantic"]

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
        overlap_sentences: int = 1,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 50,    # Discard chunks smaller than this
    ):
        assert strategy in self.STRATEGIES, f"Strategy must be one of {self.STRATEGIES}"
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.overlap_sentences = overlap_sentences
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Chunk a single document into a list of Chunk objects.
        """
        text = doc.text.strip()
        if not text:
            return []

        # Apply the chosen strategy
        if self.strategy == "fixed":
            raw_chunks = fixed_size_chunk(text, self.chunk_size, self.overlap)

        elif self.strategy == "sentence":
            raw_chunks = sentence_chunk(text, self.chunk_size, self.overlap_sentences)

        elif self.strategy == "recursive":
            raw_chunks = recursive_chunk(text, self.chunk_size, self.overlap)

        elif self.strategy == "semantic":
            raw_chunks = semantic_chunk(text, self.chunk_size, self.similarity_threshold)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Convert raw (text, start, end) tuples into Chunk objects
        chunks = []
        for i, (chunk_text, start, end) in enumerate(raw_chunks):
            # Skip chunks that are too short
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            chunk_id = f"{doc.doc_id}_chunk_{i:04d}"
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=chunk_text.strip(),
                start_char=start,
                end_char=end,
                chunk_index=i,
                strategy=self.strategy,
                metadata={
                    **doc.metadata,
                    "chunk_size_chars": len(chunk_text),
                    "chunk_size_words": len(chunk_text.split()),
                }
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  Doc '{doc.doc_id}': {len(doc.text):,} chars → {len(chunks)} chunks")
        return all_chunks

    def get_stats(self, chunks: List[Chunk]) -> Dict:
        """Compute statistics about the chunk set."""
        if not chunks:
            return {}
        sizes = [c.char_count for c in chunks]
        word_counts = [c.word_count for c in chunks]
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(sizes),
            "avg_chunk_chars": sum(sizes) / len(sizes),
            "min_chunk_chars": min(sizes),
            "max_chunk_chars": max(sizes),
            "avg_chunk_words": sum(word_counts) / len(word_counts),
            "strategy": self.strategy,
        }


# ---------------------------------------------------------------------------
# TEXT CLEANERS
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic text cleaning before chunking.
    - Normalize whitespace
    - Remove non-printable characters
    - Normalize quotes and dashes
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable characters (keep newlines and tabs)
    text = re.sub(r'[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]', '', text)

    # Normalize multiple whitespace (but preserve paragraph breaks)
    text = re.sub(r'[ \t]+', ' ', text)          # Multiple spaces/tabs → one space
    text = re.sub(r'\n{3,}', '\n\n', text)        # 3+ newlines → paragraph break

    # Normalize quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # Normalize dashes
    text = text.replace('\u2014', ' — ').replace('\u2013', ' - ')

    return text.strip()


def load_text_file(filepath: str, doc_id: Optional[str] = None) -> Document:
    """Load a plain text file as a Document."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    text = clean_text(text)
    filename = os.path.basename(filepath)
    return Document(
        doc_id=doc_id or filename,
        text=text,
        metadata={"source": filepath, "filename": filename}
    )


def load_from_string(text: str, doc_id: str = "doc_001", **metadata) -> Document:
    """Create a Document from a string."""
    return Document(doc_id=doc_id, text=clean_text(text), metadata=metadata)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_TEXT = """
    Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
    with language model generation. It was introduced by Lewis et al. in 2020 and has become
    one of the most widely used approaches for grounding language models in external knowledge.

    The core idea is simple. Instead of relying solely on knowledge stored in model parameters,
    RAG retrieves relevant documents from an external corpus at inference time. These retrieved
    documents are then provided as context to the language model alongside the original query.
    This allows the model to generate answers that are grounded in retrieved evidence.

    RAG has several important advantages over standard fine-tuning. First, the knowledge base
    can be updated without retraining the model. Second, the model can cite its sources,
    making responses more verifiable. Third, RAG works well even for highly specific or
    niche domains where training data may be scarce.

    The retrieval component typically uses dense vector search. Documents are embedded into
    a high-dimensional vector space using an encoder model. At query time, the query is
    embedded using the same encoder, and the nearest document vectors are retrieved using
    approximate nearest neighbor search algorithms like FAISS or HNSW.

    Hybrid retrieval combines dense vector search with sparse keyword-based retrieval methods
    like BM25. The scores from both systems are fused using techniques like Reciprocal Rank
    Fusion. This typically outperforms either method alone because dense retrieval handles
    semantic similarity while sparse retrieval handles exact keyword matches.
    """ * 3

    doc = load_from_string(SAMPLE_TEXT, doc_id="rag_intro", title="Introduction to RAG")

    print("=" * 60)
    print("Testing all chunking strategies")
    print("=" * 60)
    print(f"Document: {len(doc.text):,} chars, ~{len(doc.text.split())} words\n")

    for strategy in ["fixed", "sentence", "recursive", "semantic"]:
        chunker = DocumentChunker(
            strategy=strategy,
            chunk_size=300,
            overlap=50,
            min_chunk_size=30
        )
        chunks = chunker.chunk_document(doc)
        stats = chunker.get_stats(chunks)

        print(f"--- {strategy.upper()} ---")
        print(f"  Chunks:    {stats['total_chunks']}")
        print(f"  Avg chars: {stats['avg_chunk_chars']:.0f}")
        print(f"  Min/Max:   {stats['min_chunk_chars']}/{stats['max_chunk_chars']}")
        print(f"  Sample chunk 0: '{chunks[0].text[:80]}...'")
        print()

    # Full pipeline test
    print("--- Full pipeline test (recursive) ---")
    chunker = DocumentChunker(strategy="recursive", chunk_size=400, overlap=80)
    docs = [
        load_from_string("RAG combines retrieval with generation. " * 20, doc_id="doc1"),
        load_from_string("Dense embeddings capture semantic meaning. " * 20, doc_id="doc2"),
    ]
    all_chunks = chunker.chunk_documents(docs)
    print(f"\nTotal chunks from {len(docs)} documents: {len(all_chunks)}")
    print(f"First chunk: '{all_chunks[0].text[:100]}'")
    print(f"Chunk metadata: {all_chunks[0].metadata}")
