"""Shared utilities for the Lab 9 CoNLL-2003 experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

IGNORE_INDEX = -100
DEFAULT_DATASET_ID = "tner/conll2003"
DEFAULT_MODEL_ID = "distilbert/distilbert-base-cased"


@dataclass
class NerSplit:
    words: list[list[str]]
    labels: list[list[str]]


def prediction_ready(path: str | Path = "prediction.md") -> tuple[bool, str]:
    """Require a student-authored prediction before formal experiments."""
    prediction_path = Path(path)
    if not prediction_path.is_file():
        return False, f"Prediction file is missing: {prediction_path}"
    text = prediction_path.read_text(encoding="utf-8").strip()
    if not text:
        return False, "Prediction file is empty"
    if re.search(r"\bTODO\b", text, flags=re.IGNORECASE):
        return False, "Prediction file still contains TODO placeholders"
    return True, "Prediction file appears complete"


def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _label_names(dataset_split: Any, label_key: str) -> list[str]:
    feature = dataset_split.features[label_key]
    if hasattr(feature, "feature"):
        feature = feature.feature
    names = getattr(feature, "names", None)
    if not names:
        raise ValueError(f"Dataset field {label_key!r} has no ClassLabel.names metadata")
    return list(names)


def _normalise_tag(value: Any, names: list[str]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and 0 <= value < len(names):
        return names[value]
    raise ValueError(f"Cannot map dataset label {value!r} using metadata {names!r}")


def load_local_conll(directory: str | Path) -> tuple[dict[str, NerSplit], list[str]]:
    """Load standard CoNLL token-per-line files (token ... BIO-tag)."""
    root = Path(directory)
    aliases = {"train": ["train.txt", "eng.train"], "validation": ["validation.txt", "valid.txt", "eng.testa"], "test": ["test.txt", "eng.testb"]}
    splits: dict[str, NerSplit] = {}
    labels_seen: set[str] = set()
    for split, candidates in aliases.items():
        path = next((root / name for name in candidates if (root / name).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing CoNLL {split} file in {root}; accepted names: {candidates}")
        sentences: list[list[str]] = []
        tags: list[list[str]] = []
        words: list[str] = []
        labels: list[str] = []
        with path.open(encoding="utf-8") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if words:
                        sentences.append(words); tags.append(labels)
                        words, labels = [], []
                    continue
                columns = line.split()
                if len(columns) < 2:
                    raise ValueError(f"Malformed CoNLL row in {path}: {raw_line!r}")
                words.append(columns[0])
                labels.append(columns[-1])
                labels_seen.add(columns[-1])
        if words:
            sentences.append(words); tags.append(labels)
        splits[split] = NerSplit(sentences, tags)
    ordered = ["O"] + sorted(label for label in labels_seen if label != "O")
    return splits, ordered


def load_conll2003(local_dir: str | Path | None = None, dataset_id: str = DEFAULT_DATASET_ID):
    """Load authentic CoNLL-2003; local official-format files take precedence."""
    if local_dir is not None:
        return load_local_conll(local_dir)
    from datasets import load_dataset
    try:
        raw = load_dataset(dataset_id)
    except Exception as exc:
        # datasets v4 no longer executes Hub dataset scripts. The TNER mirror
        # also publishes the same CoNLL-2003 splits as JSONL plus label.json.
        if dataset_id != DEFAULT_DATASET_ID:
            raise RuntimeError(
                f"Could not load {dataset_id!r}. Supply the official CoNLL-2003 files with --data-dir. "
                "No substitute dataset is used. Original error: " + repr(exc)
            ) from exc
        try:
            import json
            from huggingface_hub import hf_hub_download
            remote_files = {
                "train": hf_hub_download(dataset_id, "dataset/train.json", repo_type="dataset"),
                "validation": hf_hub_download(dataset_id, "dataset/valid.json", repo_type="dataset"),
                "test": hf_hub_download(dataset_id, "dataset/test.json", repo_type="dataset"),
            }
            label_path = hf_hub_download(dataset_id, "dataset/label.json", repo_type="dataset")
            with open(label_path, encoding="utf-8") as stream:
                label_mapping = json.load(stream)
            label_names_override = [name for name, _ in sorted(label_mapping.items(), key=lambda pair: pair[1])]
            raw = load_dataset("json", data_files=remote_files)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Could not load CoNLL-2003 ({dataset_id}) from Hub JSONL. Supply the official local files "
                f"with --data-dir. Original dataset-script error: {exc!r}; JSON fallback error: {fallback_exc!r}"
            ) from fallback_exc
    else:
        label_names_override = None
    split_map = {"train": "train", "validation": "validation", "valid": "validation", "dev": "validation", "test": "test"}
    normalised: dict[str, NerSplit] = {}
    label_names: list[str] | None = None
    for source_name, target_name in split_map.items():
        if source_name not in raw or target_name in normalised:
            continue
        sample = raw[source_name]
        word_key = next((key for key in ("tokens", "words") if key in sample.column_names), None)
        tag_key = next((key for key in ("ner_tags", "tags", "labels") if key in sample.column_names), None)
        if word_key is None or tag_key is None:
            raise ValueError(f"Unexpected CoNLL columns in {source_name}: {sample.column_names}")
        try:
            label_names = _label_names(sample, tag_key)
        except (ValueError, KeyError):
            if label_names_override is None:
                raise
            label_names = label_names_override
        words_rows, label_rows = [], []
        for row in sample:
            words = list(row[word_key])
            tags = [_normalise_tag(value, label_names) for value in row[tag_key]]
            if len(words) != len(tags):
                raise ValueError(f"word/tag length mismatch in {source_name}: {len(words)} != {len(tags)}")
            words_rows.append([str(word) for word in words])
            label_rows.append(tags)
        normalised[target_name] = NerSplit(words_rows, label_rows)
    needed = {"train", "validation", "test"}
    missing = needed - normalised.keys()
    if missing:
        raise ValueError(f"CoNLL dataset is missing required splits: {sorted(missing)}")
    if not label_names:
        raise ValueError("Dataset did not provide label-name metadata")
    return normalised, label_names


def validate_bio_sequences(sequences: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """Report IOB2 errors without silently repairing the gold annotation."""
    errors = []
    for sentence_id, sequence in enumerate(sequences):
        previous_prefix, previous_type = "O", "O"
        for token_index, tag in enumerate(sequence):
            if tag == "O":
                prefix, entity_type = "O", "O"
            elif "-" in tag:
                prefix, entity_type = tag.split("-", 1)
            else:
                errors.append({"sentence_id": sentence_id, "token_index": token_index, "tag": tag, "reason": "tag has no BIO prefix"})
                previous_prefix, previous_type = "O", "O"
                continue
            if prefix not in {"O", "B", "I"}:
                errors.append({"sentence_id": sentence_id, "token_index": token_index, "tag": tag, "reason": "unknown BIO prefix"})
            elif prefix == "I" and not (previous_prefix in {"B", "I"} and previous_type == entity_type):
                errors.append({"sentence_id": sentence_id, "token_index": token_index, "tag": tag, "reason": "I tag does not continue same entity"})
            previous_prefix, previous_type = prefix, entity_type
    return errors


def extract_bio_spans(tags: Sequence[str], strict: bool = True) -> set[tuple[str, int, int]]:
    """Return (type, start, end-exclusive) spans using strict IOB2 semantics."""
    spans: set[tuple[str, int, int]] = set()
    start = None
    active_type = None
    for index, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            prefix, entity_type = "O", None
        elif "-" in tag:
            prefix, entity_type = tag.split("-", 1)
        else:
            raise ValueError(f"Invalid BIO label {tag!r} at token {index}")
        if prefix not in {"O", "B", "I"}:
            raise ValueError(f"Invalid BIO prefix in {tag!r} at token {index}")
        continuation = prefix == "I" and active_type == entity_type and start is not None
        if prefix == "I" and not continuation and strict:
            raise ValueError(f"Invalid IOB2 transition at token {index}: {tag}")
        if start is not None and not continuation:
            spans.add((str(active_type), start, index))
            start, active_type = None, None
        if prefix in {"B", "I"} and not continuation:
            start, active_type = index, entity_type
    return spans


def entity_counts(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], entity_types: Iterable[str] | None = None) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError(f"sentence count mismatch: {len(gold)} != {len(pred)}")
    all_gold, all_pred = [], []
    for sent_id, (gold_tags, pred_tags) in enumerate(zip(gold, pred)):
        if len(gold_tags) != len(pred_tags):
            raise ValueError(f"token count mismatch in sentence {sent_id}: {len(gold_tags)} != {len(pred_tags)}")
        # Gold annotations are checked strictly. Invalid predicted I-tags are
        # counted separately by callers and interpreted as a new span here.
        all_gold.extend((sent_id, *span) for span in extract_bio_spans(gold_tags, strict=True))
        all_pred.extend((sent_id, *span) for span in extract_bio_spans(pred_tags, strict=False))
    gold_set, pred_set = set(all_gold), set(all_pred)
    types = sorted(set(entity_types or []) | {row[1] for row in all_gold + all_pred})
    by_type = {}
    for label in types:
        g = {item for item in gold_set if item[1] == label}
        p = {item for item in pred_set if item[1] == label}
        tp = len(g & p); fp = len(p - g); fn = len(g - p)
        by_type[label] = {"tp": tp, "fp": fp, "fn": fn, "support": len(g), **_prf(tp, fp, fn)}
    tp, fp, fn = len(gold_set & pred_set), len(pred_set - gold_set), len(gold_set - pred_set)
    return {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn), "per_type": by_type, "gold_spans": gold_set, "pred_spans": pred_set}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else (0.0 if precision == recall == 0 else None)
    return {"precision": precision, "recall": recall, "f1": f1}


def entity_length_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    if len(gold) != len(pred):
        raise ValueError("gold/prediction sentence count mismatch")
    buckets = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, None, "4+")]
    output = []
    for low, high, name in buckets:
        g_all, p_all = set(), set()
        for sid, (g_tags, p_tags) in enumerate(zip(gold, pred)):
            g_all.update((sid, *s) for s in extract_bio_spans(g_tags, strict=True) if s[2]-s[1] >= low and (high is None or s[2]-s[1] <= high))
            p_all.update((sid, *s) for s in extract_bio_spans(p_tags, strict=False) if s[2]-s[1] >= low and (high is None or s[2]-s[1] <= high))
        tp = len(g_all & p_all); fp = len(p_all - g_all); fn = len(g_all - p_all)
        output.append({"entity_length": name, "gold_count": len(g_all), "predicted_count": len(p_all), "tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)})
    return output


def dataset_summary(splits: dict[str, NerSplit]) -> list[dict[str, Any]]:
    rows = []
    for name, split in splits.items():
        tag_counts = Counter(tag for row in split.labels for tag in row)
        entity_counts_by_type = Counter(span[0] for row in split.labels for span in extract_bio_spans(row, strict=False))
        invalid = validate_bio_sequences(split.labels)
        rows.append({"split": name, "sentences": len(split.words), "words": sum(map(len, split.words)), "entity_counts": dict(entity_counts_by_type), "bio_label_counts": dict(tag_counts), "invalid_bio_sequences": len(invalid)})
    return rows


def align_words(words: list[str], labels: list[str], tokenizer: Any, label2id: dict[str, int], max_length: int = 128) -> dict[str, Any]:
    if len(words) != len(labels):
        raise ValueError("word/label length mismatch")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for word_ids() alignment")
    encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, add_special_tokens=True)
    word_ids = encoded.word_ids()
    aligned = []
    seen: set[int] = set()
    for word_id in word_ids:
        if word_id is None or word_id in seen:
            aligned.append(IGNORE_INDEX)
        else:
            aligned.append(label2id[labels[word_id]])
            seen.add(word_id)
    kept = sorted(seen)
    return {"encoded": encoded, "word_ids": word_ids, "aligned_labels": aligned, "kept_word_ids": kept,
            "truncated_word_ids": sorted(set(range(len(words))) - seen),
            "split_words": {i: len([wid for wid in word_ids if wid == i]) for i in kept if word_ids.count(i) > 1}}


def align_batch(batch: dict[str, list], tokenizer: Any, label2id: dict[str, int], max_length: int = 128) -> dict[str, list]:
    encoded = tokenizer(batch["words"], is_split_into_words=True, truncation=True, max_length=max_length)
    out = {key: list(value) for key, value in encoded.items()}
    out["labels"] = []
    out["word_ids"] = []
    out["truncated_word_ids"] = []
    for batch_index, gold_labels in enumerate(batch["label_names"]):
        word_ids = encoded.word_ids(batch_index=batch_index)
        aligned, seen = [], set()
        for word_id in word_ids:
            if word_id is None or word_id in seen:
                aligned.append(IGNORE_INDEX)
            else:
                aligned.append(label2id[gold_labels[word_id]])
                seen.add(word_id)
        out["labels"].append(aligned)
        out["word_ids"].append(word_ids)
        out["truncated_word_ids"].append(sorted(set(range(len(gold_labels))) - seen))
    return out


def reconstruct_word_labels(word_ids: Sequence[int | None], token_label_ids: Sequence[int], label_names: Sequence[str], original_word_count: int | None = None) -> tuple[list[int], list[int]]:
    """Keep the first subword prediction and return corresponding word indices."""
    predictions, indices, seen = [], [], set()
    for word_id, label_id in zip(word_ids, token_label_ids):
        if word_id is not None and word_id not in seen:
            if label_id < 0 or label_id >= len(label_names):
                raise ValueError(f"Invalid predicted class id {label_id}")
            predictions.append(int(label_id)); indices.append(int(word_id)); seen.add(word_id)
    if original_word_count is not None and indices and indices != list(range(indices[0], min(original_word_count, indices[-1] + 1))):
        raise ValueError("reconstructed word positions are not contiguous; inspect truncation/alignment")
    return indices, predictions


def labels_to_ids(sequences: Sequence[Sequence[str]], label2id: dict[str, int]) -> list[list[int]]:
    return [[label2id[label] for label in seq] for seq in sequences]


def tokenization_statistics(words: list[str], tokenizer: Any, labels: list[str] | None = None) -> dict[str, Any]:
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False, truncation=False)
    word_ids = encoded.word_ids()
    counts = Counter(wid for wid in word_ids if wid is not None)
    split_ids = {wid for wid, count in counts.items() if count > 1}
    entity_ids = set()
    if labels is not None:
        entity_ids = {i for i, tag in enumerate(labels) if tag != "O"}
    return {"word_count": len(words), "subword_token_count": len(word_ids), "split_word_count": len(split_ids),
            "split_ratio": len(split_ids) / len(words) if words else 0.0,
            "R": len(word_ids) / len(words) if words else 0.0,
            "entity_word_count": len(entity_ids), "entity_split_count": len(entity_ids & split_ids),
            "non_entity_word_count": len(set(range(len(words))) - entity_ids),
            "non_entity_split_count": len(split_ids - entity_ids),
            "split_words": [words[i] for i in sorted(split_ids)]}


def find_context_type_examples(split: NerSplit, minimum: int = 3) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sid, (words, labels) in enumerate(zip(split.words, split.labels)):
        for entity_type, start, end in extract_bio_spans(labels, strict=False):
            surface = " ".join(words[start:end])
            key = " ".join(surface.casefold().split())
            groups[key].append({"surface": surface, "sentence_id": sid, "sentence": " ".join(words), "type": entity_type})
    examples = []
    for key, occurrences in groups.items():
        type_groups = defaultdict(list)
        for item in occurrences: type_groups[item["type"]].append(item)
        types = list(type_groups)
        if len(types) > 1:
            left = type_groups[types[0]][0]; right = type_groups[types[1]][0]
            examples.append({"surface_form": key, "sentence_a_id": left["sentence_id"], "sentence_a": left["sentence"], "type_a": left["type"], "sentence_b_id": right["sentence_id"], "sentence_b": right["sentence"], "type_b": right["type"]})
        if len(examples) >= minimum: break
    return examples

