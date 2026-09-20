"""LAB 01 - Core TF-IDF implementation.

The core functions below intentionally do not use sklearn's TfidfVectorizer.
They use only Python's standard library so that every step is visible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence


TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Lowercase and tokenize a document with a small word-tokenizer."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(documents: Sequence[str]) -> Dict[str, int]:
    """Return a deterministic term -> column-index mapping."""
    terms = sorted({term for doc in documents for term in tokenize(doc)})
    return {term: index for index, term in enumerate(terms)}


def compute_counts(documents: Sequence[str], vocabulary: Mapping[str, int]) -> List[List[int]]:
    """Build a dense count matrix; rows are documents and columns are terms."""
    matrix: List[List[int]] = []
    for document in documents:
        counts = Counter(tokenize(document))
        matrix.append([counts.get(term, 0) for term, _ in sorted(vocabulary.items(), key=lambda item: item[1])])
    return matrix


def compute_tf(counts: Sequence[Sequence[int]]) -> List[List[float]]:
    """Normalize each document row by its total number of tokens."""
    tf: List[List[float]] = []
    for row in counts:
        total = sum(row)
        tf.append([value / total if total else 0.0 for value in row])
    return tf


def compute_idf(counts: Sequence[Sequence[int]], smooth: bool = False) -> List[float]:
    """Compute IDF column-wise.

    smooth=False follows the lab formula log(N / df), so terms in every
    document have IDF 0. smooth=True is the common alternative
    log((1 + N)/(1 + df)) + 1.
    """
    n_documents = len(counts)
    if not n_documents:
        return []
    n_terms = len(counts[0])
    document_frequency = [sum(1 for row in counts if row[col] > 0) for col in range(n_terms)]
    if smooth:
        return [math.log((1 + n_documents) / (1 + df)) + 1.0 for df in document_frequency]
    return [math.log(n_documents / df) if df else 0.0 for df in document_frequency]


def compute_tfidf(tf: Sequence[Sequence[float]], idf: Sequence[float]) -> List[List[float]]:
    """Multiply TF and IDF element by element."""
    return [[tf_value * idf_value for tf_value, idf_value in zip(row, idf)] for row in tf]


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """Return cosine similarity, safely handling zero vectors."""
    dot = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a * a for a in x))
    norm_y = math.sqrt(sum(b * b for b in y))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return dot / (norm_x * norm_y)


def fit_tfidf(documents: Sequence[str], smooth: bool = False):
    """Fit the small TF-IDF pipeline and return all intermediate values."""
    vocabulary = build_vocabulary(documents)
    counts = compute_counts(documents, vocabulary)
    tf = compute_tf(counts)
    idf = compute_idf(counts, smooth=smooth)
    tfidf = compute_tfidf(tf, idf)
    return vocabulary, counts, tf, idf, tfidf


def transform_query(query: str, vocabulary: Mapping[str, int], idf: Sequence[float]) -> List[float]:
    """Transform a query using an existing vocabulary and IDF vector."""
    counts = Counter(tokenize(query))
    row = [counts.get(term, 0) for term, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    tf = compute_tf([row])[0]
    return [tf_value * idf_value for tf_value, idf_value in zip(tf, idf)]


def rank_documents(query: str, documents: Sequence[str], vocabulary, idf, tfidf, top_k: int = 5):
    """Return (zero-based document index, score) pairs sorted by score."""
    query_vector = transform_query(query, vocabulary, idf)
    scored = [(index, cosine_similarity(query_vector, vector)) for index, vector in enumerate(tfidf)]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def _assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    assert abs(actual - expected) < tolerance, (actual, expected)


def run_unit_tests() -> None:
    """Small tests required by the lab brief."""
    documents = ["cat eats fish", "dog eats fish", "cat likes fish"]
    vocabulary, counts, tf, idf, tfidf = fit_tfidf(documents)
    assert list(vocabulary) == ["cat", "dog", "eats", "fish", "likes"]
    assert counts == [[1, 0, 1, 1, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 1]]
    _assert_close(tf[0][vocabulary["cat"]], 1 / 3)
    _assert_close(idf[vocabulary["fish"]], 0.0)
    _assert_close(cosine_similarity([1, 1, 1], [1, 1, 0]), 2 / math.sqrt(6))
    assert transform_query("unknownword", vocabulary, idf) == [0.0] * len(vocabulary)
    print("All core unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
