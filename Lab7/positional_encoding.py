"""LAB 07, Bài 11–12: positional encoding và permutation experiment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 42
D_MODEL = 4
OUTPUT_DIR = Path(__file__).resolve().parent


def make_positional_encoding(length: int, d_model: int = D_MODEL) -> np.ndarray:
    """Tạo sinusoidal positional encoding có shape (length, d_model)."""
    if length < 1 or d_model < 2 or d_model % 2 != 0:
        raise ValueError("length phải > 0 và d_model phải là số chẵn >= 2")

    positions = np.arange(length, dtype=np.float64)[:, None]
    dimensions = np.arange(0, d_model, 2, dtype=np.float64)
    angles = positions / np.power(10000.0, dimensions / d_model)

    encoding = np.zeros((length, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)
    return encoding


def make_shared_embeddings(tokens: list[str], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Tạo một vector cho mỗi token để token giữ cùng embedding ở mọi câu."""
    return {token: rng.normal(0.0, 1.0, D_MODEL) for token in tokens}


def embed(tokens: list[str], embedding_table: dict[str, np.ndarray]) -> np.ndarray:
    """Đổi token sequence thành ma trận (số token, embedding dimension)."""
    return np.vstack([embedding_table[token] for token in tokens])


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    """Tính softmax theo từng hàng, có trừ max để ổn định số."""
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def scaled_dot_product_self_attention(
    inputs: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
) -> dict[str, np.ndarray]:
    """Tính Q, K, V, scores, weights và output cho một sequence."""
    query = inputs @ w_q
    key = inputs @ w_k
    value = inputs @ w_v
    scores = (query @ key.T) / np.sqrt(key.shape[-1])
    weights = stable_softmax(scores)
    output = weights @ value
    return {
        "Q": query,
        "K": key,
        "V": value,
        "scores": scores,
        "weights": weights,
        "output": output,
    }


def print_matrix(title: str, matrix: np.ndarray, decimals: int = 5) -> None:
    """In tên, shape và các giá trị của một ma trận."""
    print(f"\n{title} | shape={matrix.shape}")
    print(np.array2string(matrix, precision=decimals, suppress_small=True))


def run_attention(title: str, tokens: list[str], inputs: np.ndarray,
                  weights: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Chạy attention, in các ma trận và kiểm tra shape/normalization."""
    result = scaled_dot_product_self_attention(inputs, *weights)
    print(f"\n--- {title}: {tokens} ---")
    for name in ("Q", "K", "V", "scores", "weights", "output"):
        print_matrix(name, result[name])

    length = len(tokens)
    assert result["Q"].shape == (length, D_MODEL)
    assert result["K"].shape == (length, D_MODEL)
    assert result["V"].shape == (length, D_MODEL)
    assert result["scores"].shape == (length, length)
    assert result["weights"].shape == (length, length)
    assert result["output"].shape == (length, D_MODEL)
    assert np.allclose(result["weights"].sum(axis=1), 1.0)
    return result


def save_positional_encoding_figure(
    pe: np.ndarray,
    tokens_1: list[str],
    embeddings_1: np.ndarray,
    tokens_2: list[str],
    embeddings_2: np.ndarray,
) -> Path:
    """Vẽ PE, embedding gốc và embedding sau khi cộng PE cho hai sequence."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6), constrained_layout=True)
    panels = [
        (pe, ["pos 0", "pos 1", "pos 2"], "Sinusoidal positional encoding"),
        (embeddings_1, tokens_1, "Embedding — sequence 1"),
        (embeddings_1 + pe, tokens_1, "Embedding + PE — sequence 1"),
        (embeddings_2, tokens_2, "Embedding — sequence 2"),
        (embeddings_2 + pe, tokens_2, "Embedding + PE — sequence 2"),
    ]
    for axis, (matrix, row_labels, title) in zip(axes, panels):
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Embedding dimension")
        axis.set_ylabel("Position / token")
        axis.set_xticks(range(D_MODEL), labels=[f"d{i}" for i in range(D_MODEL)])
        axis.set_yticks(range(len(row_labels)), labels=row_labels)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    path = OUTPUT_DIR / "positional_encoding.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_permutation_figure(
    results: dict[str, dict[str, np.ndarray]],
    tokens_s1: list[str],
    tokens_s2: list[str],
) -> Path:
    """Vẽ bốn attention maps trên cùng thang màu [0, 1]."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], results["S1 without PE"]["weights"], tokens_s1, "S1 — without PE"),
        (axes[0, 1], results["S2 without PE"]["weights"], tokens_s2, "S2 — without PE"),
        (axes[1, 0], results["S1 with PE"]["weights"], tokens_s1, "S1 — with PE"),
        (axes[1, 1], results["S2 with PE"]["weights"], tokens_s2, "S2 — with PE"),
    ]
    image = None
    for axis, matrix, tokens, title in panels:
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xlabel("Key token")
        axis.set_ylabel("Query token")
        axis.set_xticks(range(len(tokens)), labels=tokens)
        axis.set_yticks(range(len(tokens)), labels=tokens)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Attention weight", shrink=0.82)

    path = OUTPUT_DIR / "permutation_experiment.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def experiment_1(
    embedding_table: dict[str, np.ndarray],
    pe: np.ndarray,
    attention_weights: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Thực hiện Bài 11 và in representation theo vị trí/lẫn theo token."""
    tokens_1 = ["I", "love", "NLP"]
    tokens_2 = ["NLP", "love", "I"]
    emb_1 = embed(tokens_1, embedding_table)
    emb_2 = embed(tokens_2, embedding_table)

    print("\n================ BÀI 11: POSITIONAL ENCODING ================")
    print_matrix("Embedding sequence 1: I love NLP", emb_1)
    print_matrix("Embedding sequence 2: NLP love I", emb_2)
    print_matrix("Sinusoidal positional encoding", pe)
    print_matrix("Sequence 1 without PE", emb_1)
    print_matrix("Sequence 2 without PE", emb_2)
    print_matrix("Sequence 1 with PE", emb_1 + pe)
    print_matrix("Sequence 2 with PE", emb_2 + pe)

    # So sánh theo vị trí: hai ma trận có token khác nhau ở vị trí 0 và 2.
    print("\nSo sánh theo vị trí (L2 distance giữa các hàng cùng vị trí):")
    print("without PE:", np.linalg.norm(emb_1 - emb_2, axis=1))
    print("with PE:   ", np.linalg.norm((emb_1 + pe) - (emb_2 + pe), axis=1))

    # Căn chỉnh token giống nhau trước khi so representation của chính token đó.
    positions_1 = {token: index for index, token in enumerate(tokens_1)}
    positions_2 = {token: index for index, token in enumerate(tokens_2)}
    print("\nSo sánh sau khi căn chỉnh cùng token:")
    for token in tokens_1:
        vector_1 = emb_1[positions_1[token]]
        vector_2 = emb_2[positions_2[token]]
        with_pe_1 = (emb_1 + pe)[positions_1[token]]
        with_pe_2 = (emb_2 + pe)[positions_2[token]]
        print(
            f"{token:>5}: without PE L2={np.linalg.norm(vector_1-vector_2):.8f}; "
            f"with PE L2={np.linalg.norm(with_pe_1-with_pe_2):.8f}"
        )

    results = {}
    for name, tokens, embeddings in (
        ("Sequence 1 without PE", tokens_1, emb_1),
        ("Sequence 2 without PE", tokens_2, emb_2),
        ("Sequence 1 with PE", tokens_1, emb_1 + pe),
        ("Sequence 2 with PE", tokens_2, emb_2 + pe),
    ):
        results[name] = run_attention(name, tokens, embeddings, attention_weights)
    return emb_1, emb_2


def experiment_2(
    embedding_table: dict[str, np.ndarray],
    pe: np.ndarray,
    attention_weights: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Thực hiện Bài 12, kiểm tra permutation khi có và không có PE."""
    tokens_s1 = ["dog", "bites", "man"]
    tokens_s2 = ["man", "bites", "dog"]
    permutation = np.array([2, 1, 0])
    embeddings_s1 = embed(tokens_s1, embedding_table)
    embeddings_s2 = embed(tokens_s2, embedding_table)

    print("\n================ BÀI 12: PERMUTATION EXPERIMENT ================")
    results = {}
    inputs = {
        "S1 without PE": (tokens_s1, embeddings_s1),
        "S2 without PE": (tokens_s2, embeddings_s2),
        "S1 with PE": (tokens_s1, embeddings_s1 + pe),
        "S2 with PE": (tokens_s2, embeddings_s2 + pe),
    }
    for name, (tokens, matrix) in inputs.items():
        results[name] = run_attention(name, tokens, matrix, attention_weights)

    # Không PE: S2 là phép hoán vị hàng của S1; output cũng phải hoán vị tương ứng.
    out_s1 = results["S1 without PE"]["output"]
    out_s2 = results["S2 without PE"]["output"]
    aligned_l2_without = float(np.linalg.norm(out_s2 - out_s1[permutation]))
    pooled_s1 = out_s1.mean(axis=0)
    pooled_s2 = out_s2.mean(axis=0)
    pooled_l2_without = float(np.linalg.norm(pooled_s2 - pooled_s1))
    equivariant_without = bool(np.allclose(out_s2, out_s1[permutation], atol=1e-10, rtol=1e-10))
    invariant_without = bool(np.allclose(pooled_s1, pooled_s2, atol=1e-10, rtol=1e-10))

    print("\n--- Kiểm tra without PE ---")
    print(f"L2(output_S2, output_S1[p]): {aligned_l2_without:.12e}")
    print(f"np.allclose sau căn chỉnh: {equivariant_without}")
    print(f"L2(mean_pool_S1, mean_pool_S2): {pooled_l2_without:.12e}")
    print(f"np.allclose mean pooling: {invariant_without}")
    assert equivariant_without
    assert invariant_without

    # Có PE: mỗi vị trí nhận PE cố định, nên đầu vào S2 không còn là hoán vị đơn thuần của S1.
    out_s1_pe = results["S1 with PE"]["output"]
    out_s2_pe = results["S2 with PE"]["output"]
    aligned_l2_with = float(np.linalg.norm(out_s2_pe - out_s1_pe[permutation]))
    pooled_s1_pe = out_s1_pe.mean(axis=0)
    pooled_s2_pe = out_s2_pe.mean(axis=0)
    pooled_l2_with = float(np.linalg.norm(pooled_s2_pe - pooled_s1_pe))
    equivariant_with = bool(np.allclose(out_s2_pe, out_s1_pe[permutation], atol=1e-10, rtol=1e-10))
    invariant_with = bool(np.allclose(pooled_s1_pe, pooled_s2_pe, atol=1e-10, rtol=1e-10))

    print("\n--- Kiểm tra with PE ---")
    print(f"L2(output_S2, output_S1[p]): {aligned_l2_with:.12e}")
    print(f"np.allclose sau căn chỉnh: {equivariant_with}")
    print(f"L2(mean_pool_S1, mean_pool_S2): {pooled_l2_with:.12e}")
    print(f"np.allclose mean pooling: {invariant_with}")
    print("Các giá trị trên được đo từ cùng embedding và projection weights; không gán trước kết luận.")

    return results


def main() -> None:
    """Tạo dữ liệu tái lập, chạy hai bài, lưu hình và kiểm tra các điều kiện."""
    rng = np.random.default_rng(SEED)
    all_tokens = ["I", "love", "NLP", "dog", "bites", "man"]
    embedding_table = make_shared_embeddings(all_tokens, rng)
    # Các projection weights được sinh đúng một lần và dùng chung cho mọi sequence.
    attention_weights = tuple(
        rng.normal(0.0, 0.5, size=(D_MODEL, D_MODEL)) for _ in range(3)
    )
    pe = make_positional_encoding(length=3, d_model=D_MODEL)

    assert pe.shape == (3, 4)
    assert np.array_equal(pe[0], np.array([0.0, 1.0, 0.0, 1.0]))
    assert all(vector.shape == (D_MODEL,) for vector in embedding_table.values())
    assert np.allclose(pe[0], [np.sin(0), np.cos(0), np.sin(0), np.cos(0)])

    emb_1, emb_2 = experiment_1(embedding_table, pe, attention_weights)
    permutation_results = experiment_2(embedding_table, pe, attention_weights)

    positional_path = save_positional_encoding_figure(
        pe, ["I", "love", "NLP"], emb_1,
        ["NLP", "love", "I"], emb_2,
    )
    permutation_path = save_permutation_figure(
        permutation_results, ["dog", "bites", "man"], ["man", "bites", "dog"]
    )

    assert positional_path.is_file() and positional_path.stat().st_size > 0
    assert permutation_path.is_file() and permutation_path.stat().st_size > 0
    print("\n================ TODO (STUDENT INTERPRETATION) ================")
    print("- Vì sao self-attention không có PE không phân biệt được hai thứ tự khi dùng mean pooling?")
    print("- PE thay đổi attention patterns như thế nào trong số liệu vừa in?")
    print("- Vì sao token order quan trọng?")
    print("- Permutation equivariance khác permutation invariance như thế nào?")
    print("\nĐã lưu hình:")
    print(positional_path)
    print(permutation_path)
    print("PASS: tất cả shape, softmax, PE và permutation assertions đã đạt.")


if __name__ == "__main__":
    main()
