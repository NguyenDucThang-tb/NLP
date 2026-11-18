"""Training/evaluation helpers for the Lab 9 token-classification notebook."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

from ner_core import (
    IGNORE_INDEX, NerSplit, align_words, entity_counts, entity_length_metrics,
    extract_bio_spans, reconstruct_word_labels, seed_everything, validate_bio_sequences,
)


def encode_split(split: NerSplit, tokenizer: Any, label2id: dict[str, int], max_length: int):
    examples = []
    for sid, (words, tags) in enumerate(zip(split.words, split.labels)):
        aligned = align_words(words, tags, tokenizer, label2id, max_length)
        encoded = aligned["encoded"]
        examples.append({
            "sentence_id": sid,
            "words": words,
            "gold_tags": tags,
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "labels": aligned["aligned_labels"],
            "word_ids": aligned["word_ids"],
            "kept_word_ids": aligned["kept_word_ids"],
            "truncated_word_ids": aligned["truncated_word_ids"],
        })
    return examples


def truncation_report(examples: list[dict[str, Any]]) -> dict[str, float | int]:
    total = sum(len(row["words"]) for row in examples)
    kept = sum(len(row["kept_word_ids"]) for row in examples)
    trunc_sentences = sum(bool(row["truncated_word_ids"]) for row in examples)
    return {"sentences": len(examples), "original_words": total, "evaluated_words": kept,
            "truncated_words": total - kept, "truncated_sentences": trunc_sentences,
            "word_retention_percent": 100.0 * kept / total if total else 0.0}


def _make_collator(tokenizer):
    from transformers import DataCollatorForTokenClassification
    base = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX)
    def collate(rows):
        features = [{key: row[key] for key in ("input_ids", "attention_mask", "labels")} for row in rows]
        return base(features)
    return collate


def evaluate_examples(model, examples: list[dict[str, Any]], tokenizer, label_names: list[str], batch_size: int = 8, device=None, compute_loss: bool = False):
    import torch
    from torch.utils.data import DataLoader
    device = device or next(model.parameters()).device
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=_make_collator(tokenizer))
    model.eval()
    predicted_rows: list[list[str]] = []
    gold_rows: list[list[str]] = []
    loss_total = 0.0; loss_weight = 0
    with torch.inference_mode():
        cursor = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"] if compute_loss else None)
            logits = output.logits
            if compute_loss:
                loss_total += float(output.loss.cpu()) * int(batch["input_ids"].shape[0])
                loss_weight += int(batch["input_ids"].shape[0])
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            for local_index, token_ids in enumerate(pred_ids):
                row = examples[cursor + local_index]
                first_indices, first_labels = reconstruct_word_labels(row["word_ids"], token_ids, label_names)
                if first_indices != row["kept_word_ids"]:
                    raise AssertionError("prediction reconstruction and alignment kept different word positions")
                predicted_rows.append([label_names[index] for index in first_labels])
                gold_rows.append([row["gold_tags"][index] for index in first_indices])
            cursor += len(pred_ids)
    if len(predicted_rows) != len(examples):
        raise AssertionError("evaluation lost sentence rows")
    if compute_loss:
        return gold_rows, predicted_rows, loss_total / max(loss_weight, 1)
    return gold_rows, predicted_rows


def evaluate_tags(gold_rows, pred_rows, entity_types):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    if any(len(g) != len(p) for g, p in zip(gold_rows, pred_rows)):
        raise ValueError("gold/prediction word counts differ after alignment")
    flat_gold = [tag for row in gold_rows for tag in row]
    flat_pred = [tag for row in pred_rows for tag in row]
    precision, recall, f1, _ = precision_recall_fscore_support(flat_gold, flat_pred, average="micro", zero_division=0)
    exact = entity_counts(gold_rows, pred_rows, entity_types)
    output = {"token_precision_micro": float(precision), "token_recall_micro": float(recall),
              "token_f1_micro": float(f1), "token_accuracy": float(accuracy_score(flat_gold, flat_pred)),
              "entity_precision_micro": exact["precision"], "entity_recall_micro": exact["recall"],
              "entity_f1_micro": exact["f1"], "entity_tp": exact["tp"], "entity_fp": exact["fp"],
              "entity_fn": exact["fn"], "per_type": exact["per_type"],
              "length_metrics": entity_length_metrics(gold_rows, pred_rows),
              "invalid_predicted_bio": len(validate_bio_sequences(pred_rows)),
              "invalid_gold_bio": len(validate_bio_sequences(gold_rows))}
    try:
        from seqeval.metrics import f1_score as seqeval_f1
        output["seqeval_entity_f1_micro"] = float(seqeval_f1(gold_rows, pred_rows, scheme=None, zero_division=0))
        output["custom_seqeval_f1_abs_diff"] = abs(output["entity_f1_micro"] - output["seqeval_entity_f1_micro"])
    except ImportError:
        output["seqeval_entity_f1_micro"] = None
        output["custom_seqeval_f1_abs_diff"] = None
    return output


def train_transformer(
    train_examples: list[dict[str, Any]], validation_examples: list[dict[str, Any]],
    tokenizer, label_names: list[str], model_name: str, output_dir: str | Path,
    epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5,
    weight_decay: float = 0.01, seed: int = 42, max_steps: int | None = None,
    max_length: int = 256,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForTokenClassification

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label2id = {name: index for index, name in enumerate(label_names)}
    id2label = {index: name for name, index in label2id.items()}
    config = AutoConfig.from_pretrained(model_name, num_labels=len(label_names), label2id=label2id, id2label=id2label)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model.to(device)
    collate = _make_collator(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    best_f1 = -1.0
    output = Path(output_dir); best_dir = output / "checkpoints" / "best"; best_dir.mkdir(parents=True, exist_ok=True)
    started_total = time.perf_counter(); total_steps = 0
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = 0.0; step_count = 0
        epoch_start = time.perf_counter()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()); step_count += 1; total_steps += 1
            if max_steps is not None and total_steps >= max_steps:
                break
        gold_rows, pred_rows, val_loss = evaluate_examples(model, validation_examples, tokenizer, label_names, batch_size, device, compute_loss=True)
        val_metrics = evaluate_tags(gold_rows, pred_rows, sorted({name.split("-", 1)[-1] for name in label_names if name != "O"}))
        row = {"epoch": epoch, "train_loss": loss_sum / max(step_count, 1), "validation_loss": val_loss,
               "validation_entity_f1": val_metrics["entity_f1_micro"], "validation_entity_precision": val_metrics["entity_precision_micro"],
               "validation_entity_recall": val_metrics["entity_recall_micro"], "training_seconds": time.perf_counter() - epoch_start,
               "steps": step_count, "seqeval_entity_f1": val_metrics["seqeval_entity_f1_micro"]}
        history.append(row)
        if row["validation_entity_f1"] is not None and row["validation_entity_f1"] > best_f1:
            best_f1 = row["validation_entity_f1"]
            model.save_pretrained(best_dir); tokenizer.save_pretrained(best_dir)
            with (output / "best_validation_metrics.json").open("w", encoding="utf-8") as stream:
                json.dump(val_metrics, stream, indent=2)
        print(json.dumps(row))
        if max_steps is not None and total_steps >= max_steps:
            break
    elapsed = time.perf_counter() - started_total
    result = {"model": "Pretrained Transformer token classification", "base_model": model_name,
              "seed": seed, "epochs_requested": epochs, "epochs_completed": len(history), "batch_size": batch_size,
              "learning_rate": learning_rate, "weight_decay": weight_decay, "device": str(device),
              "training_time_seconds": elapsed, "steps": total_steps, "validation_entity_f1": best_f1,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in model.parameters())}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    with (output / "training_config.json").open("w", encoding="utf-8") as stream:
        json.dump({**result, "max_length": max_length, "dataset": "tner/conll2003"}, stream, indent=2)
    return model, result, history


def save_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    import pandas as pd
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_model_predictions(split: NerSplit, gold_rows, pred_rows, path: str | Path, kept_indices=None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for sid, (words, gold, pred) in enumerate(zip(split.words, gold_rows, pred_rows)):
            indices = kept_indices[sid] if kept_indices is not None else list(range(len(words)))
            stream.write(json.dumps({"sentence_id": sid, "words": [words[i] for i in indices],
                                     "original_word_indices": indices, "gold": gold, "prediction": pred}, ensure_ascii=False) + "\n")

