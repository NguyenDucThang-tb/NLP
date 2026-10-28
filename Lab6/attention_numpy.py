"""NumPy helpers cho thí nghiệm attention cơ bản của LAB 06."""

import numpy as np


def stable_softmax(scores, axis=-1):
    """Tính softmax ổn định số bằng cách trừ giá trị lớn nhất."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("scores không được rỗng")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores phải là các giá trị hữu hạn")

    shifted_scores = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


def dot_product_attention(query, keys):
    """Tính attention chưa scaling cho một query và T key vectors.

    Args:
        query: Mảng shape (d_k,).
        keys: Mảng shape (T, d_k), mỗi hàng là một key.

    Returns:
        scores: Mảng shape (T,), tính bằng keys @ query.
        weights: Softmax của scores, shape (T,).
    """
    query = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)

    if query.ndim != 1:
        raise ValueError("query phải là vector một chiều")
    if keys.ndim != 2:
        raise ValueError("keys phải là ma trận hai chiều")
    if keys.shape[0] == 0 or keys.shape[1] != query.shape[0]:
        raise ValueError("keys phải có dạng (T, d_k) tương thích với query")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(keys)):
        raise ValueError("query và keys phải chứa các giá trị hữu hạn")

    scores = keys @ query  # Mỗi key được nhân vô hướng với query.
    weights = stable_softmax(scores, axis=0)
    return scores, weights


def self_attention(embeddings, w_query, w_key, w_value):
    """Tính scaled dot-product self-attention cho một chuỗi token.

    Mỗi hàng của embeddings tương ứng với một token. Hàm trả về Q, K, V,
    scores, attention weights và contextual representations.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    w_query = np.asarray(w_query, dtype=np.float64)
    w_key = np.asarray(w_key, dtype=np.float64)
    w_value = np.asarray(w_value, dtype=np.float64)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings phải có dạng (T, d_model) và T > 0")
    if any(not np.all(np.isfinite(x)) for x in (embeddings, w_query, w_key, w_value)):
        raise ValueError("embeddings và projection matrices phải hữu hạn")
    if w_query.ndim != 2 or w_key.ndim != 2 or w_value.ndim != 2:
        raise ValueError("projection matrices phải là ma trận hai chiều")
    if embeddings.shape[1] != w_query.shape[0] or embeddings.shape[1] != w_key.shape[0]:
        raise ValueError("dimension của embeddings không khớp W_Q hoặc W_K")
    if embeddings.shape[1] != w_value.shape[0]:
        raise ValueError("dimension của embeddings không khớp W_V")
    if w_query.shape[1] != w_key.shape[1]:
        raise ValueError("Q và K phải có cùng d_k")

    query = embeddings @ w_query
    key = embeddings @ w_key
    value = embeddings @ w_value
    scores = (query @ key.T) / np.sqrt(key.shape[1])
    weights = stable_softmax(scores, axis=-1)
    context = weights @ value
    return query, key, value, scores, weights, context
