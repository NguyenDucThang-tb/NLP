"""Extract evidence for LAB 07 Bài 26 from the trained 4-head mean classifier."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformer_experiments import (
    D_MODEL,
    D_FF,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    N_BLOCKS,
    SEED,
    SPLIT_SEED,
    ExperimentRunner,
    set_seed,
    tokenize,
)


def _select_six(samples: pd.DataFrame) -> list[int]:
    """Chọn TN, TP, FP, FN trước; sau đó lấy mẫu đúng/sai đầu tiên còn thiếu."""
    truth = samples["true_label"].to_numpy(dtype=np.int64)
    pred = samples["predicted_label"].to_numpy(dtype=np.int64)
    correct = np.flatnonzero(truth == pred).tolist()
    incorrect = np.flatnonzero(truth != pred).tolist()
    selected: list[int] = []

    # Giữ tối đa một mẫu cho mỗi ô confusion matrix trước khi bổ sung.
    for actual, predicted in ((0, 0), (1, 1), (0, 1), (1, 0)):
        candidates = [i for i in range(len(samples)) if truth[i] == actual and pred[i] == predicted]
        if candidates:
            selected.append(candidates[0])

    correct_selected = sum(truth[i] == pred[i] for i in selected)
    incorrect_selected = len(selected) - correct_selected
    for index in correct:
        if correct_selected >= 3:
            break
        if index not in selected:
            selected.append(index)
            correct_selected += 1
    for index in incorrect:
        if incorrect_selected >= 3:
            break
        if index not in selected:
            selected.append(index)
            incorrect_selected += 1

    if correct_selected < 3 or incorrect_selected < 3:
        raise RuntimeError("Test split không có đủ 3 đúng và 3 sai để phân tích")
    return selected


def _get_runner(runner: ExperimentRunner | None) -> ExperimentRunner:
    if runner is not None:
        return runner
    return ExperimentRunner(Path(__file__).resolve().parent, run_mode="full")


def _load_or_train_model(
    runner: ExperimentRunner,
    allow_retrain: bool,
) -> tuple[torch.nn.Module, dict, str]:
    lab_dir = Path(__file__).resolve().parent
    checkpoint_path = lab_dir / "checkpoints" / "transformer_heads4_mean.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    device = runner.device

    if "heads_4" in runner.models:
        model = runner.models["heads_4"]
        source = "live runner.models['heads_4'] from this notebook session"
        checkpoint = None
    elif checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_config = {
            "model": "Transformer 4 heads, masked mean pooling",
            "d_model": D_MODEL,
            "n_heads": 4,
            "d_ff": D_FF,
            "n_blocks": N_BLOCKS,
            "max_length": MAX_LENGTH,
            "pooling": "mean",
            "seed": SEED,
            "split_seed": SPLIT_SEED,
        }
        for key, value in expected_config.items():
            if checkpoint["config"].get(key) != value:
                raise ValueError(f"Checkpoint config mismatch for {key}")
        if checkpoint["vocab"] != runner.vocab:
            raise ValueError("Checkpoint vocabulary does not match LAB 05 preprocessing")
        model = runner._new_transformer(n_heads=4, max_length=MAX_LENGTH, pooling="mean")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device)
        source = str(checkpoint_path)
    elif allow_retrain:
        # Đây chỉ train lại model mục tiêu, không chạy lại các experiment khác.
        set_seed(SEED)
        model = runner._new_transformer(n_heads=4, max_length=MAX_LENGTH, pooling="mean")
        model, val_metrics, training_time, _, best_epoch = runner._train(
            model, MAX_LENGTH, "error_analysis_retrained_heads4"
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "vocab": runner.vocab,
            "config": {
                "model": "Transformer 4 heads, masked mean pooling",
                "d_model": D_MODEL,
                "n_heads": 4,
                "d_ff": D_FF,
                "n_blocks": N_BLOCKS,
                "max_length": MAX_LENGTH,
                "pooling": "mean",
                "seed": SEED,
                "split_seed": SPLIT_SEED,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "vocabulary_size": len(runner.vocab),
                "train_size": len(runner.train_ids),
                "validation_size": len(runner.val_ids),
                "test_size": len(runner.test_labels),
                "best_epoch_by_validation_f1": best_epoch,
                "validation_metrics_at_selected_checkpoint": val_metrics,
                "training_time_seconds_including_validation": training_time,
                "note": "Single-model retraining for Error Analysis; old model checkpoint was absent.",
            },
        }
        torch.save(checkpoint, checkpoint_path)
        source = f"new single-model retraining saved to {checkpoint_path}"
    else:
        raise FileNotFoundError(
            f"No live heads_4 model or checkpoint found at {checkpoint_path}. "
            "Set allow_retrain=True to train only this target model."
        )

    model = model.to(device)
    model.eval()
    if "heads_4" in runner.models:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "vocab": runner.vocab,
            "config": {
                "model": "Transformer 4 heads, masked mean pooling",
                "d_model": D_MODEL,
                "n_heads": 4,
                "d_ff": D_FF,
                "n_blocks": N_BLOCKS,
                "max_length": MAX_LENGTH,
                "pooling": "mean",
                "seed": SEED,
                "split_seed": SPLIT_SEED,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "vocabulary_size": len(runner.vocab),
                "train_size": len(runner.train_ids),
                "validation_size": len(runner.val_ids),
                "test_size": len(runner.test_labels),
                "note": "Checkpoint exported from live notebook model.",
            },
        }
        torch.save(checkpoint, checkpoint_path)
        source = f"live runner model saved to {checkpoint_path}"
    elif checkpoint_path.is_file() and checkpoint is None:
        # The model was just retrained above and its checkpoint was saved there.
        pass
    elif checkpoint_path.is_file() and checkpoint is not None:
        pass
    else:
        raise RuntimeError("Could not establish model source/checkpoint")
    return model, checkpoint or torch.load(checkpoint_path, map_location=device, weights_only=False), source


def run_error_analysis(
    runner: ExperimentRunner | None = None,
    allow_retrain: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Predict test reviews, select six deterministically, and extract attention evidence."""
    runner = _get_runner(runner)
    model, checkpoint, model_source = _load_or_train_model(runner, allow_retrain)
    device = runner.device
    test_loader = runner._loader("test", MAX_LENGTH, shuffle=False)

    labels, predictions, probabilities = [], [], []
    model.eval()
    with torch.inference_mode():
        for input_ids, batch_labels in test_loader:
            logits = model(input_ids.to(device, non_blocking=True))
            batch_probability = torch.softmax(logits, dim=1).cpu().numpy()
            labels.extend(batch_labels.numpy().tolist())
            predictions.extend(batch_probability.argmax(axis=1).tolist())
            probabilities.extend(batch_probability.tolist())
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    metrics = runner._classification_metrics(labels, predictions)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    confusion_counts = {
        "TN": int(confusion[0, 0]),
        "FP": int(confusion[0, 1]),
        "FN": int(confusion[1, 0]),
        "TP": int(confusion[1, 1]),
    }

    results_path = Path(__file__).resolve().parent / "results.csv"
    saved = pd.read_csv(results_path)
    reference = saved.loc[
        (saved["experiment"] == "Final test evaluation")
        & (saved["model"] == "Transformer (4 heads, mean)")
        & (saved["eval_split"] == "test")
    ]
    if reference.empty:
        raise ValueError("results.csv has no reference row for Transformer 4-head test")
    reference = reference.iloc[0]
    metric_match = {
        key: bool(np.isclose(metrics[key], float(reference[key]), atol=1e-6, rtol=0))
        for key in ("accuracy", "precision", "recall", "f1")
    }
    print("Fresh predictions metrics:", metrics)
    print("Fresh predictions confusion counts:", confusion_counts)
    print("Saved test metrics:", {key: float(reference[key]) for key in metric_match})
    print("Metric match:", metric_match)
    print("Model source:", model_source)

    records = []
    all_texts = runner.test_texts
    for test_index, text in enumerate(all_texts):
        tokens = tokenize(text)
        retained = tokens[:MAX_LENGTH]
        unknown = [token for token in retained if token not in runner.vocab]
        truth = int(labels[test_index])
        predicted = int(predictions[test_index])
        records.append({
            "test_index": test_index,
            "text": text,
            "true_label": truth,
            "true_name": "Positive" if truth == 1 else "Negative",
            "predicted_label": predicted,
            "predicted_name": "Positive" if predicted == 1 else "Negative",
            "p_negative": float(probabilities[test_index, 0]),
            "p_positive": float(probabilities[test_index, 1]),
            "confidence_predicted_class": float(probabilities[test_index, predicted]),
            "correct": bool(truth == predicted),
            "original_token_count": len(tokens),
            "retained_token_count": len(retained),
            "truncated_token_count": max(0, len(tokens) - MAX_LENGTH),
            "was_truncated": len(tokens) > MAX_LENGTH,
            "unk_count_in_retained": len(unknown),
            "unk_tokens_in_retained": "; ".join(unknown),
            "tokenized_input": " ".join(retained),
        })
    all_predictions = pd.DataFrame(records)
    selected_indices = _select_six(all_predictions)
    selected = all_predictions.iloc[selected_indices].copy().reset_index(drop=True)
    selected["selected_outcome"] = np.where(selected["correct"], "Correct", "Incorrect")

    # Surface flags là gợi ý để người học rà soát, không khẳng định nguyên nhân.
    negation = {"not", "no", "never", "neither", "nor", "hardly", "isn't", "wasn't", "don't", "didn't"}
    contrast = {"but", "however", "although", "though", "yet", "despite", "nevertheless"}
    flags = []
    for row in selected.to_dict("records"):
        tokens = row["tokenized_input"].split()
        text = row["text"]
        flags.append({
            "test_index": row["test_index"],
            "negation_cues": "; ".join(token for token in tokens if token in negation),
            "contrast_cues": "; ".join(token for token in tokens if token in contrast),
            "long_sequence": row["original_token_count"] > MAX_LENGTH,
            "truncated": row["was_truncated"],
            "unk_tokens": row["unk_tokens_in_retained"],
            "noise_or_markup_candidate": bool(re.search(r"<[^>]+>|[^\x00-\x7f]", text)),
        })
    selected = selected.merge(pd.DataFrame(flags), on="test_index", how="left", validate="one_to_one")

    attention_rows = []
    plot_matrices = []
    token_sequences = []
    for row in selected.to_dict("records"):
        tokens = row["tokenized_input"].split()
        ids = [runner.vocab.get(token, 1) for token in tokens]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.inference_mode():
            _, attention = model(
                input_ids, attention_mask=attention_mask, return_attention=True
            )
        matrices = attention[0, :, :len(tokens), :len(tokens)].cpu().numpy()
        plot_matrices.append(matrices.mean(axis=0))
        token_sequences.append(tokens)
        for head_index, matrix in enumerate(matrices, start=1):
            mean_received = matrix.mean(axis=0)
            top_positions = np.argsort(-mean_received, kind="stable")[:5]
            for rank, position in enumerate(top_positions, start=1):
                attention_rows.append({
                    "test_index": row["test_index"],
                    "outcome": row["selected_outcome"],
                    "head": head_index,
                    "rank": rank,
                    "token_index": int(position),
                    "token": tokens[int(position)],
                    "mean_attention_received": float(mean_received[position]),
                    "is_surface_sentiment_cue": tokens[int(position)] in {
                        "good", "great", "excellent", "amazing", "love", "loved", "enjoyed",
                        "bad", "awful", "terrible", "worst", "hate", "boring", "poor", "not",
                    },
                })

    attention_evidence = pd.DataFrame(attention_rows)
    lab_dir = Path(__file__).resolve().parent
    samples_path = lab_dir / "error_samples.csv"
    attention_path = lab_dir / "error_attention_evidence.csv"
    selected.to_csv(samples_path, index=False)
    attention_evidence.to_csv(attention_path, index=False)
    figure_path = lab_dir / "error_attention_examples.png"

    fig, axes = plt.subplots(3, 2, figsize=(16, 16), constrained_layout=True)
    vmax = max(float(matrix.max()) for matrix in plot_matrices)
    image = None
    for axis, row, matrix, tokens in zip(axes.flat, selected.to_dict("records"), plot_matrices, token_sequences):
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(
            f"Test {row['test_index']} — {row['selected_outcome']} — "
            f"true {row['true_name']} / pred {row['predicted_name']}\n"
            "Mean attention over 4 heads"
        )
        axis.set_xlabel("Key token index")
        axis.set_ylabel("Query token index")
        stride = max(1, int(np.ceil(len(tokens) / 16)))
        positions = np.arange(0, len(tokens), stride)
        labels_at_positions = [f"{i}:{tokens[i]}" for i in positions]
        axis.set_xticks(positions, labels=labels_at_positions, rotation=90, fontsize=6)
        axis.set_yticks(positions, labels=labels_at_positions, fontsize=6)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Mean attention weight", shrink=0.75)
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model_source": model_source,
        "checkpoint": str(lab_dir / "checkpoints" / "transformer_heads4_mean.pt"),
        "metrics": metrics,
        "confusion_counts": confusion_counts,
        "reference_metrics": {key: float(reference[key]) for key in metric_match},
        "metrics_match": metric_match,
        "metrics_match_all": all(metric_match.values()),
        "selected_correct": int(selected["correct"].sum()),
        "selected_incorrect": int((~selected["correct"]).sum()),
        "selected_test_indices": selected["test_index"].astype(int).tolist(),
        "artifacts": [str(samples_path), str(attention_path), str(figure_path)],
    }
    print("Selected examples:")
    print(selected[[
        "test_index", "true_name", "predicted_name", "p_negative", "p_positive",
        "original_token_count", "retained_token_count", "truncated_token_count", "unk_count_in_retained",
    ]].to_string(index=False))
    print("Saved:", samples_path, attention_path, figure_path)
    return selected, attention_evidence, summary


if __name__ == "__main__":
    # Explicit opt-in: first run can retrain this target model only if no checkpoint exists.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()
    run_error_analysis(allow_retrain=args.train_if_missing)
