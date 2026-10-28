"""Data and artifact helpers for LAB 06 Experiment 3."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attention_numpy import self_attention


SENTENCE1 = ["The", "cat", "that", "I", "adopted", "yesterday", "is", "sleeping"]
SENTENCE2 = ["The", "book", "that", "I", "bought", "yesterday", "was", "expensive"]
SELECTED_TOKENS = ["cat", "adopted", "sleeping"]


def create_synthetic_parameters(seed=42, d_model=4, d_k=4, d_v=4):
    """Create shared token embeddings and Q/K/V projections with a fixed seed."""
    unique_tokens = list(dict.fromkeys(SENTENCE1 + SENTENCE2))
    rng = np.random.default_rng(seed)
    embeddings = {
        token: rng.normal(size=d_model).astype(np.float64)
        for token in unique_tokens
    }
    w_query = rng.normal(size=(d_model, d_k)).astype(np.float64)
    w_key = rng.normal(size=(d_model, d_k)).astype(np.float64)
    w_value = rng.normal(size=(d_model, d_v)).astype(np.float64)
    return embeddings, w_query, w_key, w_value


def _compute_sentence(tokens, embeddings, w_query, w_key, w_value):
    x = np.vstack([embeddings[token] for token in tokens])
    return x, self_attention(x, w_query, w_key, w_value)


def _save_matrix_csv(path, tokens, matrix):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Query token / Key token", *tokens])
        for token, row in zip(tokens, matrix):
            writer.writerow([token, *(f"{value:.12f}" for value in row)])


def _save_heatmap(path, tokens, matrix, title):
    fig, ax = plt.subplots(figsize=(9, 7))
    heat = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(tokens)), tokens, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(tokens)), tokens)
    ax.set_xlabel("Key tokens (attended to)")
    ax.set_ylabel("Query tokens (attending)")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] < 0.5 else "black"
            ax.text(
                col, row, f"{matrix[row, col]:.3f}",
                ha="center", va="center", color=color, fontsize=8,
            )
    fig.colorbar(heat, ax=ax, label="Attention Weight")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment3(output_dir):
    """Generate both self-attention matrices, selected-token CSV and figures."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    if len(SENTENCE1) != 8 or len(SENTENCE2) != 8:
        raise AssertionError("Both sentence token lists must contain 8 tokens")

    embeddings, w_query, w_key, w_value = create_synthetic_parameters()
    x1, output1 = _compute_sentence(SENTENCE1, embeddings, w_query, w_key, w_value)
    x2, output2 = _compute_sentence(SENTENCE2, embeddings, w_query, w_key, w_value)

    selected_rows = []
    for source_token in SELECTED_TOKENS:
        source_index = SENTENCE1.index(source_token)
        row_weights = output1[4][source_index]
        top_indices = np.argsort(-row_weights, kind="stable")[:3]
        for rank, target_index in enumerate(top_indices, start=1):
            selected_rows.append(
                (source_token, source_index, SENTENCE1[target_index],
                 float(row_weights[target_index]), rank)
            )

    files = {
        "heatmap_sentence1": output_dir / "experiment3_attention_heatmap_sentence1.png",
        "heatmap_sentence2": output_dir / "experiment3_attention_heatmap_sentence2.png",
        "selected_comparison": output_dir / "experiment3_selected_tokens_comparison.png",
        "selected_csv": output_dir / "experiment3_selected_token_analysis.csv",
        "matrix_sentence1_csv": output_dir / "experiment3_attention_matrix_sentence1.csv",
        "matrix_sentence2_csv": output_dir / "experiment3_attention_matrix_sentence2.csv",
    }

    _save_heatmap(
        files["heatmap_sentence1"], SENTENCE1, output1[4],
        "Experiment 3 - Self-Attention - Sentence 1",
    )
    _save_heatmap(
        files["heatmap_sentence2"], SENTENCE2, output2[4],
        "Experiment 3 - Self-Attention - Sentence 2",
    )
    _save_matrix_csv(files["matrix_sentence1_csv"], SENTENCE1, output1[4])
    _save_matrix_csv(files["matrix_sentence2_csv"], SENTENCE2, output2[4])

    with files["selected_csv"].open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Source Token", "Source Position", "Attended Token", "Weight", "Rank"])
        writer.writerows(
            (token, position, attended, f"{weight:.12f}", rank)
            for token, position, attended, weight, rank in selected_rows
        )

    x_positions = np.arange(len(SENTENCE1))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for series_index, token in enumerate(SELECTED_TOKENS):
        row_index = SENTENCE1.index(token)
        offset = (series_index - 1) * width
        ax.bar(x_positions + offset, output1[4][row_index], width, label=f"Query token: {token}")
    ax.set_xticks(x_positions, SENTENCE1, rotation=25, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Experiment 3 - Selected Token Attention Distributions")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(files["selected_comparison"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "tokens1": SENTENCE1.copy(),
        "tokens2": SENTENCE2.copy(),
        "embeddings": embeddings,
        "w_query": w_query,
        "w_key": w_key,
        "w_value": w_value,
        "x1": x1,
        "x2": x2,
        "output1": output1,
        "output2": output2,
        "selected_rows": selected_rows,
        "files": files,
    }
