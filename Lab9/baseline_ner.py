"""CoNLL-2003 token-feature baseline using DictVectorizer + LogisticRegression."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ner_core import (
    NerSplit, dataset_summary, entity_counts, entity_length_metrics,
    extract_bio_spans, load_conll2003, prediction_ready, seed_everything, validate_bio_sequences,
)


def extract_token_features(words: list[str], index: int) -> dict[str, Any]:
    """Make local lexical and shape features; no neighboring gold labels."""
    word = words[index]
    previous = words[index - 1] if index > 0 else "<BOS>"
    following = words[index + 1] if index + 1 < len(words) else "<EOS>"
    features: dict[str, Any] = {
        "word.lower": word.lower(), "word.original": word,
        "word.isupper": word.isupper(), "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(), "word.has_hyphen": "-" in word,
        "previous.lower": previous.lower(), "next.lower": following.lower(),
        "previous.istitle": previous.istitle(), "next.istitle": following.istitle(),
        "begin_sentence": index == 0, "end_sentence": index == len(words) - 1,
    }
    for size in range(1, 4):
        features[f"prefix_{size}"] = word[:size]
        features[f"suffix_{size}"] = word[-size:]
    return features


def build_feature_matrix(split: NerSplit):
    from sklearn.feature_extraction import DictVectorizer
    rows = [extract_token_features(words, index) for words in split.words for index in range(len(words))]
    vectorizer = DictVectorizer(sparse=True)
    return vectorizer.fit_transform(rows), vectorizer


def _split_feature_rows(split: NerSplit) -> list[dict[str, Any]]:
    return [extract_token_features(words, index) for words in split.words for index in range(len(words))]


def train_baseline(train: NerSplit, seed: int = 42):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    features = _split_feature_rows(train)
    labels = [tag for row in train.labels for tag in row]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(features)
    model = LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs", n_jobs=None)
    started = time.perf_counter()
    model.fit(x_train, labels)
    elapsed = time.perf_counter() - started
    return model, vectorizer, elapsed


def predict_baseline(model, vectorizer, split: NerSplit) -> list[list[str]]:
    from sklearn.preprocessing import LabelEncoder
    rows = _split_feature_rows(split)
    flat = model.predict(vectorizer.transform(rows)).tolist()
    output, cursor = [], 0
    for words in split.words:
        output.append(flat[cursor:cursor + len(words)])
        cursor += len(words)
    if cursor != len(flat):
        raise AssertionError("prediction reconstruction lost or duplicated token positions")
    return output


def evaluate_baseline(split: NerSplit, predictions: list[list[str]], entity_types: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    if [len(x) for x in split.words] != [len(x) for x in predictions]:
        raise ValueError("word-level predictions do not align with original words")
    gold_flat = [tag for row in split.labels for tag in row]
    pred_flat = [tag for row in predictions for tag in row]
    precision, recall, f1, _ = precision_recall_fscore_support(gold_flat, pred_flat, average="micro", zero_division=0)
    exact = entity_counts(split.labels, predictions, entity_types)
    invalid_pred = validate_bio_sequences(predictions)
    return {
        "token_precision_micro": float(precision), "token_recall_micro": float(recall),
        "token_f1_micro": float(f1), "token_accuracy": float(accuracy_score(gold_flat, pred_flat)),
        "entity_precision_micro": exact["precision"], "entity_recall_micro": exact["recall"],
        "entity_f1_micro": exact["f1"], "entity_tp": exact["tp"], "entity_fp": exact["fp"], "entity_fn": exact["fn"],
        "per_type": exact["per_type"], "invalid_predicted_bio_tags": len(invalid_pred),
        "invalid_gold_bio_tags": len(validate_bio_sequences(split.labels)),
        "length_metrics": entity_length_metrics(split.labels, predictions),
        "spans": exact,
    }


def _upsert_csv(path: Path, row: dict[str, Any], key: str = "model") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = list(row)
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = list(dict.fromkeys((reader.fieldnames or []) + fields))
            rows = [item for item in reader if item.get(key) != row[key]]
    rows.append({name: str(row.get(name, "")) for name in fields})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _save_confusion_matrix(gold: list[list[str]], pred: list[list[str]], labels: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    y_true = [tag for row in gold for tag in row]
    y_pred = [tag for row in pred for tag in row]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Baseline NER — token-level confusion matrix")
    ax.set_xlabel("Predicted BIO label"); ax.set_ylabel("Gold BIO label")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def _save_entity_length_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [row["entity_length"] for row in rows]
    y = [float(row["f1"]) if row["f1"] is not None else float("nan") for row in rows]
    ax.bar(x, y, color="#4c78a8")
    ax.set(title="Baseline entity-level F1 by gold entity length", xlabel="Entity length (original words)", ylabel="Strict exact-span F1", ylim=(0, 1))
    for index, row in enumerate(rows): ax.text(index, y[index] if y[index] == y[index] else 0, f'n={row["gold_count"]}', ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)


def _save_errors(split: NerSplit, predictions: list[list[str]], output: Path) -> None:
    """Export actual baseline FP/FN/boundary candidates with explanation left blank."""
    candidates = {"False positive (non-boundary)": [], "False negative (non-boundary)": [], "Boundary error": []}
    for sid, (words, gold_tags, pred_tags) in enumerate(zip(split.words, split.labels, predictions)):
        gold_spans = extract_bio_spans(gold_tags, strict=True)
        predicted_spans = extract_bio_spans(pred_tags, strict=False)
        paired_gold, paired_pred = set(), set()
        for gold_span in gold_spans:
            for predicted_span in predicted_spans:
                same_type = gold_span[0] == predicted_span[0]
                overlaps = max(gold_span[1], predicted_span[1]) < min(gold_span[2], predicted_span[2])
                exact_boundary = gold_span[1:] == predicted_span[1:]
                if same_type and overlaps and not exact_boundary:
                    paired_gold.add(gold_span); paired_pred.add(predicted_span)
                    candidates["Boundary error"].append((sid, words, gold_span, predicted_span))
        for span in sorted(predicted_spans - gold_spans - paired_pred):
            candidates["False positive (non-boundary)"].append((sid, words, None, span))
        for span in sorted(gold_spans - predicted_spans - paired_gold):
            candidates["False negative (non-boundary)"].append((sid, words, span, None))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("# Error analysis evidence — baseline\n\n")
        stream.write("Các ví dụ dưới đây là ứng viên được tính từ test predictions của baseline. Đây chưa phải lỗi của Transformer. Cột giải thích để trống cho sinh viên tự xem câu và tự phân tích.\n\n")
        stream.write(f"Số ứng viên tìm được: { {key: len(value) for key, value in candidates.items()} }\n\n")
        stream.write("| Sentence ID | Original sentence | Gold span/type/word offsets | Predicted span/type/word offsets | Error category | Student explanation |\n|---:|---|---|---|---|---|\n")
        for category, rows in candidates.items():
            for sid, words, gold_span, predicted_span in rows[:5]:
                gold_text = "—" if gold_span is None else f"{' '.join(words[gold_span[1]:gold_span[2]])} / {gold_span[0]} / [{gold_span[1]}, {gold_span[2]})"
                pred_text = "—" if predicted_span is None else f"{' '.join(words[predicted_span[1]:predicted_span[2]])} / {predicted_span[0]} / [{predicted_span[1]}, {predicted_span[2]})"
                stream.write(f"| {sid} | {' '.join(words)} | {gold_text} | {pred_text} | {category} | TODO |\n")
        stream.write("\n## Việc sinh viên cần làm\n\nTự xác nhận từng candidate, chọn ít nhất 5 FP, 5 FN và 5 boundary errors nếu có đủ. Sau đó bổ sung giải thích dựa trên câu gốc. Khi Transformer được fine-tune, chạy lại notebook để có bảng lỗi riêng cho Transformer.\n")


def run(args: argparse.Namespace) -> None:
    ready, reason = prediction_ready(args.prediction_file)
    if not ready:
        raise RuntimeError(f"Formal baseline is gated: {reason}. Complete prediction.md first.")
    seed_everything(args.seed)
    splits, label_names = load_conll2003(args.data_dir, args.dataset_id)
    summary = dataset_summary(splits)
    print("Dataset:", args.dataset_id if not args.data_dir else args.data_dir)
    print("Label metadata:", label_names)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Invalid gold BIO labels:", {name: len(validate_bio_sequences(part.labels)) for name, part in splits.items()})
    model, vectorizer, train_time = train_baseline(splits["train"], args.seed)
    predictions = predict_baseline(model, vectorizer, splits["test"])
    metrics = evaluate_baseline(splits["test"], predictions, [label.split("-", 1)[-1] for label in label_names if label != "O"])
    metrics.pop("spans")
    print("Test metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    output = Path(args.output_dir); (output / "baseline").mkdir(parents=True, exist_ok=True)
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "baseline" / "training_config.json").write_text(json.dumps({
        "dataset": args.dataset_id if not args.data_dir else str(args.data_dir),
        "train_sentences": len(splits["train"].words), "train_words": sum(map(len, splits["train"].words)),
        "seed": args.seed, "estimator": "sklearn.linear_model.LogisticRegression",
        "solver": "lbfgs", "max_iter": 500, "features": "token shape, prefixes/suffixes, previous/next lexical tokens",
        "training_time_seconds": train_time,
    }, indent=2), encoding="utf-8")
    with (output / "baseline" / "model.pkl").open("wb") as stream:
        pickle.dump({"model": model, "vectorizer": vectorizer, "labels": label_names}, stream)
    with (output / "baseline" / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for sid, (words, gold, pred) in enumerate(zip(splits["test"].words, splits["test"].labels, predictions)):
            stream.write(json.dumps({"sentence_id": sid, "words": words, "gold": gold, "prediction": pred}, ensure_ascii=False) + "\n")
    result = {"experiment": "baseline_ner", "model": "Token features + Logistic Regression", "dataset": args.dataset_id if not args.data_dir else str(args.data_dir),
              "test_sentences": len(splits["test"].words), "test_words": sum(map(len, splits["test"].words)),
              "evaluated_words": sum(map(len, splits["test"].words)), "evaluation_unit": "strict exact entity span; token scores supplementary",
              "checkpoint": str(output / "baseline" / "model.pkl"), "feature_count": len(vectorizer.vocabulary_),
              "training_time_seconds": round(train_time, 4), "status": "EXECUTED",
              **{k: v for k, v in metrics.items() if k != "per_type" and k != "length_metrics"}}
    _upsert_csv(Path(args.results), result)
    per_type_path = Path(args.entity_metrics)
    with per_type_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["model", "entity_type", "precision", "recall", "f1", "support", "tp", "fp", "fn"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for entity_type, values in metrics["per_type"].items():
            writer.writerow({"model": result["model"], "entity_type": entity_type, **values})
    _save_confusion_matrix(splits["test"].labels, predictions, label_names, Path(args.confusion_matrix))
    _save_entity_length_plot(metrics["length_metrics"], Path(args.length_plot))
    _save_errors(splits["test"], predictions, Path(args.error_analysis))
    print("Saved model, predictions, metrics and baseline plots under", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="tner/conll2003")
    parser.add_argument("--data-dir", help="Directory containing official train/validation/test CoNLL files")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--entity-metrics", default="entity_metrics.csv")
    parser.add_argument("--confusion-matrix", default="confusion_matrix.png")
    parser.add_argument("--length-plot", default="entity_length_analysis.png")
    parser.add_argument("--error-analysis", default="error_analysis.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-file", default="prediction.md")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
