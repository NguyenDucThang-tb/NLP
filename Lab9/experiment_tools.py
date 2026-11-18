"""Reproducible tables, evidence exports and figures for Lab 9."""

from __future__ import annotations

import csv
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

from ner_core import NerSplit, extract_bio_spans, find_context_type_examples, tokenization_statistics


def make_tokenization_rows(split: NerSplit, tokenizer, count: int = 20):
    eligible = [i for i, words in enumerate(split.words) if len(words) >= 5]
    if len(eligible) < count:
        raise ValueError(f"Need at least {count} sentences with >=5 original words; found {len(eligible)}")
    lengths = sorted(eligible, key=lambda i: (len(split.words[i]), i))
    short_count = count // 2
    short_ids = set(lengths[:short_count])
    selected = lengths[:short_count] + lengths[-(count - short_count):]
    rows = []
    split_counts = Counter()
    entity_words = {"count": 0, "split": 0, "subwords": 0}
    non_entity_words = {"count": 0, "split": 0, "subwords": 0}
    for sid in selected:
        words, labels = split.words[sid], split.labels[sid]
        stats = tokenization_statistics(words, tokenizer, labels)
        split_counts.update(stats["split_words"])
        rows.append({"sentence_id": sid, "length_group": "short" if sid in short_ids else "long",
                     "original_text": " ".join(words), **{key: value for key, value in stats.items() if key != "split_words"}})
        token_encoding = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        word_ids = token_encoding.word_ids()
        per_word = Counter(word_ids)
        for word_id, word in enumerate(words):
            is_entity = labels[word_id] != "O"
            aggregate = entity_words if is_entity else non_entity_words
            aggregate["count"] += 1
            aggregate["subwords"] += per_word.get(word_id, 0)
            aggregate["split"] += per_word.get(word_id, 0) > 1
    aggregate_rows = []
    for name, values in [("entity", entity_words), ("non-entity", non_entity_words)]:
        aggregate_rows.append({"word_group": name, "word_count": values["count"], "split_word_count": values["split"],
                               "mean_subwords_per_word": values["subwords"] / values["count"] if values["count"] else 0,
                               "split_ratio": values["split"] / values["count"] if values["count"] else 0})
    return rows, aggregate_rows, split_counts


def save_tokenization_outputs(rows, aggregate_rows, split_counts, output_dir: str | Path):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tokenization_analysis.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(output / "entity_vs_nonentity_tokenization.csv", index=False)
    values = frame["R"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(values, bins=10, color="#4c78a8", edgecolor="white")
    ax.set(title="CoNLL-2003 sample: subword pieces per original word", xlabel="R = subword tokens / original words", ylabel="Number of sentences")
    fig.tight_layout(); fig.savefig(output / "tokenization_ratio.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(frame["word_count"], frame["subword_token_count"], color="#f58518")
    ax.set(title="Original words and subword token counts", xlabel="# original words", ylabel="# subword tokens (special tokens excluded)")
    fig.tight_layout(); fig.savefig(output / "words_vs_subwords.png", dpi=180); plt.close(fig)
    common = split_counts.most_common(20)
    with (output / "frequently_split_words.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["word", "split_occurrences"]); writer.writerows(common)
    return frame


def write_context_analysis(split: NerSplit, path: str | Path):
    examples = find_context_type_examples(split, minimum=100)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Context analysis — CoNLL-2003 test split\n\n")
        stream.write("Các ví dụ dưới đây được trích từ gold spans của test split; chưa có phần giải thích thay sinh viên. Surface form được chuẩn hóa bằng `casefold()` và gộp whitespace để tìm trùng.\n\n")
        stream.write(f"Số surface form có nhiều entity type: **{len(examples)}**. Đề yêu cầu ít nhất 3 trường hợp nếu test set cung cấp đủ.\n\n")
        stream.write("| Surface form | Sentence A (id) | Type A | Sentence B (id) | Type B | Student analysis |\n|---|---|---|---|---|---|\n")
        for row in examples:
            stream.write(f"| {row['surface_form']} | {row['sentence_a']} ({row['sentence_a_id']}) | {row['type_a']} | {row['sentence_b']} ({row['sentence_b_id']}) | {row['type_b']} | TODO |\n")
        stream.write("\n## Câu hỏi tự phân tích\n\n- Cùng surface form xuất hiện trong ngữ cảnh nào khác nhau?\n- Context nào giúp phân biệt các nhãn gold?\n- Static representation sẽ thiếu thông tin gì trong các ví dụ này?\n")
    return examples


def _select_ie_rows(split: NerSplit, count: int = 30):
    candidates = []
    for sid, (words, tags) in enumerate(zip(split.words, split.labels)):
        spans = extract_bio_spans(tags)
        types = {span[0] for span in spans}
        if not spans:
            continue
        candidates.append((sid, words, spans, types))
    selected = []
    covered_types: set[str] = set()
    for row in candidates:
        if row[3] - covered_types:
            selected.append(row); covered_types |= row[3]
        if len(covered_types) >= 4:
            break
    for row in candidates:
        if row in selected: continue
        if any(end - start > 1 for _, start, end in row[2]) and not any(any(e - s > 1 for _, s, e in item[2]) for item in selected):
            selected.append(row)
        if len(selected) >= count: break
    for row in candidates:
        if row not in selected:
            selected.append(row)
        if len(selected) >= count: break
    return selected[:count]


def write_ie_gold_template(split: NerSplit, path: str | Path, count: int = 30):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["sentence_id", "text", "gold_entities_json", "gold_source", "manual_review_notes"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for sid, words, spans, _ in _select_ie_rows(split, count):
            char_offsets = []
            cursor = 0
            for word in words:
                char_offsets.append((cursor, cursor + len(word)))
                cursor += len(word) + 1
            entities = [{"text": " ".join(words[start:end]), "label": entity_type,
                         "start_word": start, "end_word": end,
                         "start": char_offsets[start][0], "end": char_offsets[end - 1][1]}
                        for entity_type, start, end in sorted(spans)]
            writer.writerow({"sentence_id": sid, "text": " ".join(words),
                             "gold_entities_json": json.dumps(entities, ensure_ascii=False),
                             "gold_source": "CoNLL-2003 test gold labels", "manual_review_notes": ""})


def write_environment_metadata(path: str | Path, dataset_id: str, splits: dict[str, NerSplit], model_name: str, seed: int = 42):
    import importlib.metadata as metadata
    import torch
    packages = {}
    for name in ["torch", "transformers", "datasets", "accelerate", "scikit-learn", "seqeval", "numpy", "pandas", "matplotlib"]:
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    report = {"dataset_source": dataset_id, "dataset_license_note": "See the CoNLL-2003 dataset card and original Reuters data terms.",
              "split_sizes_sentences": {name: len(part.words) for name, part in splits.items()},
              "split_sizes_words": {name: sum(map(len, part.words)) for name, part in splits.items()},
              "base_model": model_name, "random_seed": seed, "python": platform.python_version(),
              "platform": platform.platform(), "packages": packages, "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

