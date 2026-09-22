"""mấy hàm tf-idf cơ bản cho bài lab.

ở đây em không dùng tfidfvectorizer có sẵn.
em tự làm từng bước để dễ nhìn hơn.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence


# regex này dùng để lấy các từ trong câu
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str) -> List[str]:
    """đổi chữ về thường rồi tách thành các từ nhỏ."""
    # lowercase để các chữ hoa và chữ thường được tính như nhau
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(documents: Sequence[str]) -> Dict[str, int]:
    """tạo danh sách từ và đánh số vị trí cho từng từ."""
    # set để không bị lặp từ
    terms = sorted({term for doc in documents for term in tokenize(doc)})
    # enumerate để mỗi từ có một số cột
    return {term: index for index, term in enumerate(terms)}


def compute_counts(documents: Sequence[str], vocabulary: Mapping[str, int]) -> List[List[int]]:
    """đếm xem mỗi từ xuất hiện bao nhiêu lần trong từng document."""
    matrix: List[List[int]] = []
    for document in documents:
        # counter đếm số lần xuất hiện của từng từ
        counts = Counter(tokenize(document))
        matrix.append([counts.get(term, 0) for term, _ in sorted(vocabulary.items(), key=lambda item: item[1])])
    return matrix


def compute_tf(counts: Sequence[Sequence[int]]) -> List[List[float]]:
    """chia số lần xuất hiện cho tổng số từ của document."""
    tf: List[List[float]] = []
    for row in counts:
        # tổng số từ dùng để chia cho count
        total = sum(row)
        tf.append([value / total if total else 0.0 for value in row])
    return tf


def compute_idf(counts: Sequence[Sequence[int]], smooth: bool = False) -> List[float]:
    """tính idf cho từng từ.

    nếu smooth là false thì dùng công thức của bài là log(n / df).
    nếu từ có trong tất cả document thì idf của nó sẽ bằng 0.
    """
    n_documents = len(counts)
    if not n_documents:
        return []
    n_terms = len(counts[0])
    # đếm xem một từ xuất hiện trong bao nhiêu document
    document_frequency = [sum(1 for row in counts if row[col] > 0) for col in range(n_terms)]
    if smooth:
        return [math.log((1 + n_documents) / (1 + df)) + 1.0 for df in document_frequency]
    return [math.log(n_documents / df) if df else 0.0 for df in document_frequency]


def compute_tfidf(tf: Sequence[Sequence[float]], idf: Sequence[float]) -> List[List[float]]:
    """nhân tf với idf để ra tf-idf."""
    # nhân từng giá trị cùng vị trí với nhau
    return [[tf_value * idf_value for tf_value, idf_value in zip(row, idf)] for row in tf]


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """tính độ giống nhau giữa hai vector."""
    # dot là tích vô hướng của hai vector
    dot = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a * a for a in x))
    norm_y = math.sqrt(sum(b * b for b in y))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return dot / (norm_x * norm_y)


def fit_tfidf(documents: Sequence[str], smooth: bool = False):
    """chạy lần lượt các bước và trả về các kết quả trung gian."""
    # chạy từng phần theo đúng pipeline của bài
    vocabulary = build_vocabulary(documents)
    counts = compute_counts(documents, vocabulary)
    tf = compute_tf(counts)
    idf = compute_idf(counts, smooth=smooth)
    tfidf = compute_tfidf(tf, idf)
    return vocabulary, counts, tf, idf, tfidf


def transform_query(query: str, vocabulary: Mapping[str, int], idf: Sequence[float]) -> List[float]:
    """đổi query sang vector để đem đi so sánh với document."""
    # query cũng phải dùng vocabulary cũ của document
    counts = Counter(tokenize(query))
    row = [counts.get(term, 0) for term, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    tf = compute_tf([row])[0]
    return [tf_value * idf_value for tf_value, idf_value in zip(tf, idf)]


def rank_documents(query: str, documents: Sequence[str], vocabulary, idf, tfidf, top_k: int = 5):
    """xếp các document theo điểm giống nhau từ cao xuống thấp."""
    # đổi query thành vector trước khi tính điểm
    query_vector = transform_query(query, vocabulary, idf)
    scored = [(index, cosine_similarity(query_vector, vector)) for index, vector in enumerate(tfidf)]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def _assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    assert abs(actual - expected) < tolerance, (actual, expected)


def run_unit_tests() -> None:
    """test thử bằng mấy document ngắn xem hàm có chạy đúng không."""
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
