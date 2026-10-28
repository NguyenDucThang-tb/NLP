"""Application 26: inference and attention evidence from the trained E4 model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from experiment4_models import (
    AttentionClassifier,
    MAX_LENGTH,
    MAX_VOCAB_SIZE,
    PAD_ID,
    PAD_TOKEN,
    SEED,
    UNK_ID,
    UNK_TOKEN,
    encode_text,
    tokenize,
)


LAB_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = LAB_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "checkpoints" / "experiment4_attention.pt"
CONFIG_PATH = ARTIFACTS_DIR / "experiment4_config.json"

INPUTS = [
    ("Sentence 1", "I like this movie.", "Positive"),
    ("Sentence 2", "I do not like this movie.", "Negative"),
    (
        "Sentence 3",
        "I thought this movie would be bad, but it was actually excellent.",
        "Positive",
    ),
]
CHECKED_TERMS = {"like", "not", "bad", "but", "excellent"}
LABEL_TO_ID = {"Negative": 0, "Positive": 1}


def load_trained_attention(device: torch.device):
    """Restore the exact model and token-to-ID vocabulary saved by Experiment 4."""
    if not CHECKPOINT_PATH.is_file() or not CONFIG_PATH.is_file():
        raise FileNotFoundError("Experiment 4 checkpoint/configuration is missing")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("label_mapping") != {"0": "Negative", "1": "Positive"}:
        raise ValueError(f"Unexpected label mapping in E4 config: {config.get('label_mapping')}")
    if config.get("max_length") != MAX_LENGTH or config.get("vocabulary_size_including_PAD_UNK") != MAX_VOCAB_SIZE:
        raise ValueError("Preprocessing configuration differs from the E4 implementation")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "Attention":
        raise ValueError("Checkpoint is not the trained Experiment 4 Attention model")
    vocab = checkpoint.get("vocabulary")
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(vocab, dict) or len(vocab) != MAX_VOCAB_SIZE:
        raise ValueError("Checkpoint does not contain the expected full vocabulary mapping")
    if vocab.get(PAD_TOKEN) != PAD_ID or vocab.get(UNK_TOKEN) != UNK_ID:
        raise ValueError("PAD/UNK IDs do not match the Experiment 4 preprocessing")
    for key in ("max_length", "embedding_dim", "vocabulary_size"):
        expected = {
            "max_length": MAX_LENGTH,
            "embedding_dim": config["embedding_dim"],
            "vocabulary_size": len(vocab),
        }[key]
        if checkpoint_config.get(key) != expected:
            raise ValueError(f"Checkpoint config mismatch for {key}")

    model = AttentionClassifier(len(vocab)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, vocab, config, checkpoint_config


def _plot_attention(tokens: list[str], weights: np.ndarray, path: Path, title: str) -> None:
    """Plot the valid-token submatrix from the trained model's forward pass."""
    if weights.shape != (len(tokens), len(tokens)):
        raise ValueError("Heatmap tokens and attention matrix dimensions do not match")
    fig_size = max(7.0, min(18.0, 0.55 * len(tokens)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))
    image = ax.imshow(weights, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    tick_size = 9 if len(tokens) <= 20 else 6
    ax.set_xticks(np.arange(len(tokens)), tokens, rotation=45, ha="right", fontsize=tick_size)
    ax.set_yticks(np.arange(len(tokens)), tokens, fontsize=tick_size)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_one(text: str, model, vocab: dict[str, int], device: torch.device) -> dict:
    tokens = tokenize(text)
    encoded, length = encode_text(text, vocab)
    token_ids = encoded[:length]
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits, representation, attention = model(input_ids, lengths)
        logit = float(logits[0].cpu())
        p_positive = float(torch.sigmoid(logits[0]).cpu())
        matrix = attention[0, :length, :length].cpu().numpy().copy()
        forward_attention_shape = tuple(attention.shape)
        full_matrix = attention[0].cpu().numpy().copy()

    p_negative = 1.0 - p_positive
    predicted_id = int(p_positive >= 0.5)
    received = matrix.mean(axis=0)  # mean(A[:, token_index]) over valid queries only
    oov_tokens = [token for token in tokens[:MAX_LENGTH] if token not in vocab]
    trunc_count = max(0, len(tokens) - MAX_LENGTH)
    term_status = "; ".join(
        f"{term}={'NOT_IN_TEXT' if term not in tokens else ('IN_VOCAB' if term in vocab else 'OOV')}"
        for term in sorted(CHECKED_TERMS)
    )

    return {
        "Text": text,
        "Tokens": tokens,
        "Token IDs": token_ids,
        "Sequence length before truncation": len(tokens),
        "Valid sequence length": length,
        "Truncated token count": trunc_count,
        "UNK tokens": oov_tokens,
        "Checked term status": term_status,
        "Logit": logit,
        "P(Positive)": p_positive,
        "P(Negative)": p_negative,
        "Prediction ID": predicted_id,
        "Prediction": "Positive" if predicted_id == LABEL_TO_ID["Positive"] else "Negative",
        "Predicted-class probability": max(p_positive, p_negative),
        "Attention matrix shape": forward_attention_shape,
        "Valid-token attention shape": tuple(matrix.shape),
        "Attention matrix": matrix,
        "Full attention matrix": full_matrix,
        "Mean attention received": received,
        "Representation shape": tuple(representation.shape),
        "Input IDs tensor": input_ids,
        "Length tensor": lengths,
    }


def _first_test_example(model, vocab, device, want_correct: bool) -> dict:
    """Return the first matching example in original IMDb test order."""
    test_data = load_dataset("imdb")["test"].to_pandas()[["text", "label"]]
    model.eval()
    batch_size = 128
    for start in range(0, len(test_data), batch_size):
        batch = test_data.iloc[start : start + batch_size]
        encoded = [encode_text(str(text), vocab) for text in batch["text"]]
        input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
        lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long, device=device)
        truth = batch["label"].to_numpy(dtype=np.int64)
        with torch.inference_mode():
            logits, _, attention = model(input_ids, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions = (probs >= 0.5).astype(np.int64)
        matches = np.flatnonzero(predictions == truth if want_correct else predictions != truth)
        if len(matches):
            local = int(matches[0])
            global_index = start + local
            text = str(batch.iloc[local]["text"])
            result = _run_one(text, model, vocab, device)
            result.update({
                "Source": "IMDb held-out test (first matching row in dataset order)",
                "Test row index": global_index,
                "True label ID": int(truth[local]),
                "True label": "Positive" if int(truth[local]) == 1 else "Negative",
                "Correct?": bool(predictions[local] == truth[local]),
            })
            return result
    raise RuntimeError("Could not find requested correct/incorrect example in IMDb test split")


def _prediction_row(name: str, expected: str, result: dict, heatmap_path: Path) -> dict:
    expected_id = LABEL_TO_ID[expected]
    row = {
        "Sentence": name,
        "Text": result["Text"],
        "Expected": expected,
        "Prediction": result["Prediction"],
        "P(Positive)": result["P(Positive)"],
        "P(Negative)": result["P(Negative)"],
        "Predicted-class probability": result["Predicted-class probability"],
        "Correct?": result["Prediction ID"] == expected_id,
        "Tokens": " ".join(result["Tokens"]),
        "Token IDs": " ".join(map(str, result["Token IDs"])),
        "Sequence length before truncation": result["Sequence length before truncation"],
        "Valid sequence length": result["Valid sequence length"],
        "Truncated token count": result["Truncated token count"],
        "UNK tokens": " ".join(result["UNK tokens"]),
        "Checked term status": result["Checked term status"],
        "Attention matrix shape": str(result["Attention matrix shape"]),
        "Valid-token attention shape": str(result["Valid-token attention shape"]),
        "Attention heatmap": str(heatmap_path),
    }
    return row


def run_application() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, config, checkpoint_config = load_trained_attention(device)

    # Snapshot E1-E4 artifacts to assert this application leaves them untouched.
    protected_paths = sorted(ARTIFACTS_DIR.glob("experiment[1-4]_*") ) + sorted(
        (ARTIFACTS_DIR / "checkpoints").glob("experiment4_*.pt")
    )
    protected = {path.relative_to(ARTIFACTS_DIR).as_posix(): path.read_bytes() for path in protected_paths}

    sentence_results = []
    prediction_rows = []
    received_rows = []
    heatmap_paths = []
    matrix_archive = {}
    for index, (name, text, expected) in enumerate(INPUTS, start=1):
        result = _run_one(text, model, vocab, device)
        heatmap_path = ARTIFACTS_DIR / f"application26_attention_sentence{index}.png"
        _plot_attention(result["Tokens"][:MAX_LENGTH], result["Attention matrix"], heatmap_path,
                        f"Trained Attention — {name}")
        result.update({"Source": "Required prompt sentence", "Expected": expected})
        sentence_results.append(result)
        prediction_rows.append(_prediction_row(name, expected, result, heatmap_path))
        heatmap_paths.append(heatmap_path)
        matrix_archive[f"sentence{index}_attention"] = result["Attention matrix"]
        matrix_archive[f"sentence{index}_tokens"] = np.asarray(result["Tokens"][:MAX_LENGTH], dtype=str)

        ranking = np.argsort(-result["Mean attention received"], kind="stable")
        for rank, token_index in enumerate(ranking, start=1):
            received_rows.append({
                "Sentence": name,
                "Token": result["Tokens"][token_index],
                "Token position": int(token_index),
                "Mean attention received": float(result["Mean attention received"][token_index]),
                "Rank": rank,
                "Formula": "mean(A[:, token_index]) over valid query positions",
            })

    # Start with the required sentences. Add a held-out IMDb example only if needed.
    selected_correct = next((r for r in sentence_results if r["Prediction"] == r["Expected"]), None)
    selected_incorrect = next((r for r in sentence_results if r["Prediction"] != r["Expected"]), None)
    if selected_correct is None:
        selected_correct = _first_test_example(model, vocab, device, want_correct=True)
    if selected_incorrect is None:
        selected_incorrect = _first_test_example(model, vocab, device, want_correct=False)

    selected_examples = []
    selected_paths = {}
    for correct, result in ((True, selected_correct), (False, selected_incorrect)):
        kind = "correct" if correct else "incorrect"
        path = ARTIFACTS_DIR / f"application26_selected_{kind}_attention.png"
        _plot_attention(
            result["Tokens"][:MAX_LENGTH], result["Attention matrix"], path,
            f"Trained Attention — Selected {kind} example",
        )
        selected_paths[kind] = path
        expected_or_true = result.get("Expected", result.get("True label"))
        is_correct = result.get("Prediction") == expected_or_true
        selected_examples.append({
            "Source": result.get("Source", "Required prompt sentence"),
            "Text": result["Text"],
            "True/Expected": expected_or_true,
            "Predicted": result["Prediction"],
            "P(Predicted class)": result["Predicted-class probability"],
            "P(Positive)": result["P(Positive)"],
            "P(Negative)": result["P(Negative)"],
            "Correct?": bool(is_correct),
            "Tokens": " ".join(result["Tokens"]),
            "Token IDs": " ".join(map(str, result["Token IDs"])),
            "UNK tokens": " ".join(result["UNK tokens"]),
            "Truncated token count": result["Truncated token count"],
            "Attention matrix shape": str(result["Attention matrix shape"]),
            "Valid-token attention shape": str(result["Valid-token attention shape"]),
            "Attention visualization": str(path),
            "Observation data": "Use the saved matrix and tokenization as evidence; cause not assigned.",
        })
        matrix_archive[f"selected_{kind}_attention"] = result["Attention matrix"]
        matrix_archive[f"selected_{kind}_tokens"] = np.asarray(result["Tokens"][:MAX_LENGTH], dtype=str)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = ARTIFACTS_DIR / "application26_predictions.csv"
    selected_path = ARTIFACTS_DIR / "application26_selected_examples.csv"
    received_path = ARTIFACTS_DIR / "application26_attention_received.csv"
    npz_path = ARTIFACTS_DIR / "application26_attention_matrices.npz"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(selected_examples).to_csv(selected_path, index=False)
    pd.DataFrame(received_rows).to_csv(received_path, index=False)
    np.savez_compressed(npz_path, **matrix_archive)

    # Verify the protected files have not changed before reporting PASS.
    tests = []
    def check(name: str, condition) -> None:
        if not bool(condition):
            raise AssertionError(f"FAIL: {name}")
        tests.append(name)
        print(f"PASS: {name}")

    check("checkpoint loads strictly and vocabulary mapping is complete",
          len(vocab) == MAX_VOCAB_SIZE and vocab[PAD_TOKEN] == PAD_ID and vocab[UNK_TOKEN] == UNK_ID)
    check("preprocessing and labels match Experiment 4 configuration",
          config["max_length"] == MAX_LENGTH and config["label_mapping"] == {"0": "Negative", "1": "Positive"} and
          checkpoint_config["max_length"] == MAX_LENGTH and
          "not" in tokenize("I do not like this movie.") and
          encode_text("unseenapplicationtoken", vocab)[0][0] == UNK_ID)
    check("the exact three required sentences were used",
          [row["Text"] for row in sentence_results] == [row[1] for row in INPUTS])
    check("inference returned logits, probabilities and attention for all sentences",
          len(sentence_results) == 3 and all(np.isfinite(row["Logit"]) for row in sentence_results))

    probability_ok = all(
        0 <= row["P(Positive)"] <= 1 and 0 <= row["P(Negative)"] <= 1 and
        np.isclose(row["P(Positive)"] + row["P(Negative)"], 1.0, atol=1e-7) and
        row["Prediction"] == ("Positive" if row["P(Positive)"] >= row["P(Negative)"] else "Negative")
        for row in sentence_results
    )
    check("probabilities are valid and prediction matches the larger class probability", probability_ok)
    check("attention matrix shapes match the padded sequence length",
          all(row["Attention matrix shape"] == (1, MAX_LENGTH, MAX_LENGTH) and
              row["Valid-token attention shape"] == (row["Valid sequence length"],) * 2
              for row in sentence_results))

    padding_ok = True
    normalization_ok = True
    finite_ok = True
    for row in sentence_results:
        length = row["Valid sequence length"]
        full = row["Full attention matrix"]
        padding_ok &= np.all(full[:length, length:] == 0) and np.all(full[length:, :] == 0)
        normalization_ok &= np.allclose(full[:length, :length].sum(axis=1), 1.0, atol=1e-6)
        finite_ok &= np.isfinite(full).all() and np.isfinite(row["P(Positive)"]) and np.isfinite(row["Logit"])
    check("PAD keys and padded query rows are masked", padding_ok)
    check("valid attention rows sum to one", normalization_ok)
    check("logits, probabilities and attention weights are finite", finite_ok)
    check("correct/incorrect flags are computed from expected labels",
          all(row["Correct?"] == (row["Prediction"] == row["Expected"]) for row in prediction_rows))
    check("at least one actual correct and one actual incorrect example are selected",
          any(row["Correct?"] for row in selected_examples) and any(not row["Correct?"] for row in selected_examples))

    # Directly compare plotted arrays with the matrices produced in inference.
    check("heatmaps use the forward-pass attention matrices",
          all(np.array_equal(matrix_archive[f"sentence{i}_attention"], sentence_results[i - 1]["Attention matrix"])
              for i in range(1, 4)))
    repeat = [_run_one(text, model, vocab, device) for _, text, _ in INPUTS]
    check("same checkpoint and input reproduce inference outputs",
          all(np.array_equal(a["Attention matrix"], b["Attention matrix"]) and
              a["Logit"] == b["Logit"] for a, b in zip(sentence_results, repeat)))

    protected_unchanged = all((ARTIFACTS_DIR / name).read_bytes() == data for name, data in protected.items())
    check("Experiment 1-4 files and checkpoints remain byte-identical", protected_unchanged)

    output_paths = [
        predictions_path, selected_path, received_path, npz_path,
        *heatmap_paths, *selected_paths.values(),
    ]
    check("all CSV/NPZ/PNG artifacts exist and are non-empty",
          all(path.is_file() and path.stat().st_size > 0 for path in output_paths))
    predictions_readback = pd.read_csv(predictions_path)
    selected_readback = pd.read_csv(selected_path)
    check("output tables have expected rows and required columns",
          len(predictions_readback) == 3 and len(selected_readback) == 2 and
          {"Expected", "Prediction", "P(Positive)", "P(Negative)", "Correct?"}.issubset(predictions_readback.columns) and
          {"True/Expected", "Predicted", "P(Predicted class)", "Attention visualization"}.issubset(selected_readback.columns))

    report_path = ARTIFACTS_DIR / "application26_test_report.txt"
    report_path.write_text("\n".join(f"PASS: {name}" for name in tests) + "\n", encoding="utf-8")
    print("\nPredictions:")
    print(pd.DataFrame(prediction_rows)[["Sentence", "Expected", "Prediction", "P(Positive)", "P(Negative)", "Correct?"]].to_string(index=False))
    print("\nSelected examples:")
    print(pd.DataFrame(selected_examples)[["Source", "True/Expected", "Predicted", "P(Predicted class)", "Correct?"]].to_string(index=False))
    print(f"\nCheckpoint: {CHECKPOINT_PATH}\nDevice: {device}\nPassed tests: {len(tests)}")
    return {
        "predictions": prediction_rows,
        "selected_examples": selected_examples,
        "files": output_paths + [report_path],
        "tests": tests,
    }


if __name__ == "__main__":
    run_application()
