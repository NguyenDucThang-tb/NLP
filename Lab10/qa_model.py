"""Utilities and tested implementations for the LAB 10 extractive QA project.

The alignment, decoding, SQuAD metric and span-error routines are exercised by
unit tests before the notebook opens its training gate.
"""

from __future__ import annotations

import random
import re
import time
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASET_ID = "rajpurkar/squad"
MODEL_ID = "distilbert-base-uncased"
RESULT_COLUMNS = [
    "experiment", "model", "setting", "split", "num_samples", "max_length",
    "stride", "train_windows", "eval_windows", "train_answer_coverage",
    "validation_answer_coverage", "heldout_answer_coverage", "training_steps",
    "em", "f1", "training_time_seconds", "seed",
]


@dataclass(frozen=True)
class QAConfig:
    """One reproducible run configuration; values are editable in the notebook."""

    mode: str = "QUICK"
    dataset_id: str = DATASET_ID
    model_id: str = MODEL_ID
    seed: int = 42
    max_length: int = 384
    stride: int = 128
    batch_size: int = 8
    learning_rate: float = 3e-5
    epochs: int = 2
    optimizer_name: str = "AdamW"
    weight_decay: float = 0.01
    truncation: str = "only_second"
    return_overflowing_tokens: bool = True
    return_offsets_mapping: bool = True
    max_train_samples: int = 10_000
    max_validation_samples: int = 1_000
    max_test_samples: int = 1_000
    quick_train_samples: int = 64
    quick_validation_samples: int = 16
    quick_test_samples: int = 16
    max_answer_length: int = 30
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    def validate(self) -> None:
        if self.mode not in {"QUICK", "FULL"}:
            raise ValueError("mode must be QUICK or FULL")
        if self.max_length < 16:
            raise ValueError("max_length must be at least 16")
        # Tokenizer overflow stride must be smaller than the feature length.
        if self.stride < 0 or self.stride >= self.max_length:
            raise ValueError("stride must satisfy 0 <= stride < max_length")
        if self.batch_size < 1 or self.epochs < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size, epochs and gradient_accumulation_steps must be positive")
        if self.optimizer_name != "AdamW":
            raise ValueError("This implementation currently supports optimizer_name='AdamW'")

    def sample_limits(self) -> tuple[int, int, int]:
        self.validate()
        if self.mode == "QUICK":
            return self.quick_train_samples, self.quick_validation_samples, self.quick_test_samples
        return self.max_train_samples, self.max_validation_samples, self.max_test_samples

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int = 42) -> None:
    """Set Python, NumPy and Torch seeds when those packages are installed."""
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
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    """Return cuda when available, otherwise cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _balanced_title_partition(rows: Sequence[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split official SQuAD validation by article title, keeping titles disjoint."""
    title_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        title_groups.setdefault(str(row.get("title", "")), []).append(dict(row))
    if len(title_groups) < 2:
        raise ValueError("Need at least two article titles to make disjoint validation/test groups")

    titles = list(title_groups)
    random.Random(seed).shuffle(titles)
    # Greedily balance question counts without splitting questions from one article.
    ordered_titles = sorted(titles, key=lambda title: len(title_groups[title]), reverse=True)
    validation_titles: list[str] = []
    test_titles: list[str] = []
    validation_size = test_size = 0
    for title in ordered_titles:
        if validation_size <= test_size:
            validation_titles.append(title)
            validation_size += len(title_groups[title])
        else:
            test_titles.append(title)
            test_size += len(title_groups[title])
    validation_rows = [row for title in validation_titles for row in title_groups[title]]
    test_rows = [row for title in test_titles for row in title_groups[title]]
    return validation_rows, test_rows


def _sample_rows(rows: Sequence[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    copied = [dict(row) for row in rows]
    if len(copied) <= limit:
        return copied
    selected = random.Random(seed).sample(range(len(copied)), limit)
    return [copied[index] for index in selected]


def create_fixed_splits(dataset: Any, config: QAConfig) -> dict[str, list[dict[str, Any]]]:
    """Use SQuAD train, and split official validation into title-disjoint val/test."""
    train_limit, validation_limit, test_limit = config.sample_limits()
    train_rows = [dict(row) for row in dataset["train"]]
    official_validation = [dict(row) for row in dataset["validation"]]

    train_titles = {str(row.get("title", "")) for row in train_rows}
    dev_titles = {str(row.get("title", "")) for row in official_validation}
    title_overlap = train_titles & dev_titles
    if title_overlap:
        raise ValueError(f"Article title overlap between SQuAD train and validation: {sorted(title_overlap)[:5]}")

    validation_rows, test_rows = _balanced_title_partition(official_validation, config.seed)
    return {
        "train": _sample_rows(train_rows, train_limit, config.seed),
        "validation": _sample_rows(validation_rows, validation_limit, config.seed + 1),
        "test": _sample_rows(test_rows, test_limit, config.seed + 2),
    }


def validate_split_titles(splits: dict[str, Sequence[dict[str, Any]]]) -> dict[str, list[str]]:
    """Raise on duplicate IDs/article leakage and return titles for the split manifest."""
    titles = {name: {str(row.get("title", "")) for row in rows} for name, rows in splits.items()}
    ids: dict[str, set[str]] = {}
    for name, rows in splits.items():
        row_ids = [str(row.get("id", "")) for row in rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError(f"Duplicate sample IDs inside {name} split")
        ids[name] = set(row_ids)
    names = list(titles)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared_ids = ids[first] & ids[second]
            if shared_ids:
                raise ValueError(f"Sample ID leakage between {first} and {second}: {sorted(shared_ids)[:5]}")
            overlap = titles[first] & titles[second]
            if overlap:
                raise ValueError(f"Article title leakage between {first} and {second}: {sorted(overlap)[:5]}")
    return {name: sorted(values) for name, values in titles.items()}


def save_json(path: str | Path, value: Any) -> None:
    """Write reproducibility metadata as readable UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def upsert_results_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append real experiment rows while replacing an existing same-run key."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, Any]] = []
    if destination.is_file():
        with destination.open(newline="", encoding="utf-8") as stream:
            old_rows = list(csv.DictReader(stream))

    def key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(name, "")) for name in ("experiment", "model", "setting", "split", "seed"))

    new_by_key = {key(row): row for row in rows}
    merged = [row for row in old_rows if key(row) not in new_by_key]
    merged.extend(new_by_key.values())
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in RESULT_COLUMNS} for row in merged)


def environment_report(config: QAConfig, device: str | None = None) -> dict[str, Any]:
    """Collect actual runtime versions and accelerator details without requiring optional packages."""
    import platform
    info: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": device or get_device(), "dataset": config.dataset_id,
        "model": config.model_id, "seed": config.seed, "mode": config.mode,
        "config": config.to_dict(),
    }
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                     "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        if torch.cuda.is_available():
            info["gpu_memory_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        info.update({"torch": None, "cuda_available": False, "gpu_name": None})
    for package in ("transformers", "datasets", "numpy", "pandas", "matplotlib"):
        try:
            from importlib.metadata import version
            info[package] = version(package)
        except Exception:
            info[package] = None
    return info


def load_squad(dataset_id: str = DATASET_ID) -> Any:
    """Load SQuAD through Hugging Face Datasets; no substitute dataset is used."""
    from datasets import load_dataset
    return load_dataset(dataset_id)


def example_answer_is_in_context(example: dict[str, Any]) -> bool:
    """Check source character annotation consistency, not tokenizer alignment."""
    answers = example.get("answers", {})
    texts = answers.get("text", [])
    starts = answers.get("answer_start", [])
    return bool(texts and starts and example["context"][starts[0]:starts[0] + len(texts[0])] == texts[0])


def build_tokenizer(model_id: str = MODEL_ID):
    """Load a fast tokenizer needed to inspect offsets and overflow windows."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("W10 requires a fast tokenizer for offset_mapping and word_ids/sequence_ids")
    return tokenizer


def tokenizer_accepts_token_type_ids(tokenizer: Any) -> bool:
    """Check model input names rather than assuming token_type_ids exist."""
    return "token_type_ids" in getattr(tokenizer, "model_input_names", [])


def build_qa_model(model_id: str = MODEL_ID):
    """Load a pretrained encoder with a question-answering start/end head."""
    from transformers import AutoModelForQuestionAnswering
    return AutoModelForQuestionAnswering.from_pretrained(model_id)


def forward_qa_batch(model: Any, batch: dict[str, Any], device: Any):
    """Run a QA forward pass and let the model compute training loss if labels exist."""
    model_inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions") if key in batch}
    return model(**model_inputs)


def train_one_epoch(model: Any, dataloader: Any, optimizer: Any, device: Any,
                    mixed_precision: bool = True, gradient_accumulation_steps: int = 1,
                    scaler: Any = None) -> float:
    """One standard supervised epoch; input batches need student-aligned start/end labels."""
    import torch
    from tqdm.auto import tqdm

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    use_amp = bool(mixed_precision and str(device).startswith("cuda"))
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    loss_sum = 0.0
    steps = 0
    total_batches = len(dataloader)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(tqdm(dataloader, desc="train", leave=False)):
        group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(gradient_accumulation_steps, total_batches - group_start)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = forward_qa_batch(model, batch, device)
            if outputs.loss is None:
                raise ValueError("Training batch must contain start_positions and end_positions")
            scaled_loss = outputs.loss / group_size
        scaler.scale(scaled_loss).backward()
        is_update_step = (batch_index + 1) % gradient_accumulation_steps == 0 or batch_index + 1 == total_batches
        if is_update_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(outputs.loss.detach().cpu())
        steps += 1
    if steps == 0:
        raise ValueError("Training dataloader is empty")
    return loss_sum / steps


def evaluate_loss(model: Any, dataloader: Any, device: Any) -> float:
    """Compute mean validation loss from the model-provided start/end loss."""
    import torch
    model.eval()
    loss_sum = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = forward_qa_batch(model, batch, device)
            if outputs.loss is None:
                raise ValueError("Validation-loss batches need start_positions and end_positions")
            loss_sum += float(outputs.loss.detach().cpu())
            steps += 1
    if steps == 0:
        raise ValueError("Validation dataloader is empty")
    return loss_sum / steps


def train_with_validation(model: Any, train_loader: Any, validation_loader: Any, validation_fn: Any,
                          config: QAConfig, checkpoint_dir: str,
                          resume_from: str | None = None) -> tuple[Any, list[dict[str, float]], float]:
    """Train epochs and save the best model by callback-provided validation F1.

    validation_fn(model) must use the student's span decoder and EM/F1 functions.
    Test data must never be passed to this function.
    """
    import json
    from pathlib import Path
    import torch
    from tqdm.auto import tqdm

    device = torch.device(get_device())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    use_amp = bool(config.mixed_precision and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    best_validation_f1 = float("-inf")
    training_seconds = 0.0
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trainer_state_path = Path(resume_from or destination) / "trainer_state.pt"
    start_epoch = 0
    if resume_from:
        if Path(resume_from).resolve() != destination.resolve():
            raise ValueError("resume_from must be the same run directory as checkpoint_dir")
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"Resume state not found: {trainer_state_path}")
        try:
            trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        except TypeError:  # Compatibility with PyTorch versions without weights_only.
            trainer_state = torch.load(trainer_state_path, map_location="cpu")
        model.load_state_dict(trainer_state["model_state_dict"])
        optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        if "scaler_state_dict" in trainer_state:
            scaler.load_state_dict(trainer_state["scaler_state_dict"])
        history = trainer_state.get("history", [])
        start_epoch = int(trainer_state["epoch"])
        best_validation_f1 = float(trainer_state["best_validation_f1"])
        training_seconds = float(trainer_state.get("training_seconds", 0.0))
        if "python_rng_state" in trainer_state:
            random.setstate(trainer_state["python_rng_state"])
        if "numpy_rng_state" in trainer_state:
            import numpy as np
            np.random.set_state(trainer_state["numpy_rng_state"])
        if "torch_rng_state" in trainer_state:
            torch.set_rng_state(trainer_state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in trainer_state:
            torch.cuda.set_rng_state_all(trainer_state["cuda_rng_state"])
        generator = getattr(train_loader, "generator", None)
        if generator is not None and "loader_rng_state" in trainer_state:
            generator.set_state(trainer_state["loader_rng_state"])
        print(f"Resuming at epoch {start_epoch + 1}/{config.epochs}")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config.mixed_precision,
                                     config.gradient_accumulation_steps, scaler)
        training_seconds += time.perf_counter() - epoch_start
        validation_loss = evaluate_loss(model, validation_loader, device)
        validation_metrics = validation_fn(model)
        if "f1" not in validation_metrics:
            raise KeyError("validation_fn must return the student's QA token F1 under key 'f1'")
        row = {"epoch": float(epoch + 1), "train_loss": train_loss,
               "validation_loss": validation_loss, "validation_em": float(validation_metrics.get("em", float("nan"))),
               "validation_f1": float(validation_metrics["f1"])}
        history.append(row)
        if row["validation_f1"] > best_validation_f1:
            best_validation_f1 = row["validation_f1"]
            model.save_pretrained(destination)
            save_json(destination / "run_config.json", config.to_dict())
        # Resume state is saved every epoch, while the inference checkpoint remains the best one.
        import numpy as np
        generator = getattr(train_loader, "generator", None)
        torch.save({"epoch": epoch + 1, "best_validation_f1": best_validation_f1,
                    "training_seconds": training_seconds, "history": history,
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "loader_rng_state": generator.get_state() if generator is not None else None},
                   destination / "trainer_state.pt")
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
              f"val_em={row['validation_em']:.4f} val_f1={row['validation_f1']:.4f}")
    # Return the best validation-F1 weights, not necessarily the last epoch's weights.
    best_checkpoint = destination / "model.safetensors"
    if best_checkpoint.is_file():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(best_checkpoint), device=str(device)))
    else:
        best_checkpoint = destination / "pytorch_model.bin"
        if best_checkpoint.is_file():
            model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
    return model, history, training_seconds


def character_to_token_span(offset_mapping: Any, sequence_ids: Sequence[int | None], answer_start: int, answer_text: str) -> tuple[int, int] | None:
    """Map a complete end-exclusive character span to inclusive context-token indices."""
    if not answer_text or answer_start < 0 or len(offset_mapping) != len(sequence_ids):
        return None

    answer_end = answer_start + len(answer_text)
    context_offsets: list[tuple[int, int, int]] = []
    for token_index, (offset, sequence_id) in enumerate(zip(offset_mapping, sequence_ids)):
        if sequence_id != 1 or offset is None or len(offset) != 2:
            continue
        token_start, token_end = int(offset[0]), int(offset[1])
        # (0, 0) is used for special/padding tokens; empty offsets are not tokens.
        if token_start < 0 or token_end <= token_start:
            continue
        context_offsets.append((token_index, token_start, token_end))

    if not context_offsets:
        return None

    # Reject answers outside or clipped by either edge of this context window.
    window_start = min(start for _, start, _ in context_offsets)
    window_end = max(end for _, _, end in context_offsets)
    if answer_start < window_start or answer_end > window_end:
        return None

    overlapping = [
        token_index for token_index, token_start, token_end in context_offsets
        if token_start < answer_end and token_end > answer_start
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1]


def prepare_train_features(tokenizer: Any, examples: Sequence[dict[str, Any]], max_length: int, stride: int) -> list[dict[str, Any]]:
    """Tokenize train examples into context windows and attach inclusive span labels."""
    if not examples:
        return []
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for offset and sequence alignment")

    questions = [str(example["question"]).lstrip() for example in examples]
    contexts = [str(example["context"]) for example in examples]
    encoded = tokenizer(
        questions, contexts, truncation="only_second", max_length=max_length,
        stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
        padding=False,
    )
    mapping = encoded.get("overflow_to_sample_mapping")
    if mapping is None:
        raise KeyError("Tokenizer did not return overflow_to_sample_mapping")

    features: list[dict[str, Any]] = []
    for feature_index, source_index_value in enumerate(mapping):
        source_index = int(source_index_value)
        if not 0 <= source_index < len(examples):
            raise ValueError(f"Invalid overflow source index: {source_index}")
        input_ids = list(encoded["input_ids"][feature_index])
        sequence_ids = encoded.sequence_ids(feature_index)
        offsets = encoded["offset_mapping"][feature_index]
        answers = examples[source_index].get("answers", {})
        answer_texts = list(answers.get("text") or [])
        answer_starts = list(answers.get("answer_start") or [])

        cls_id = getattr(tokenizer, "cls_token_id", None)
        try:
            cls_position = input_ids.index(int(cls_id)) if cls_id is not None else 0
        except ValueError:
            cls_position = 0

        aligned_span = None
        for answer_text, answer_start in zip(answer_texts, answer_starts):
            aligned_span = character_to_token_span(offsets, sequence_ids, int(answer_start), str(answer_text))
            if aligned_span is not None:
                break

        row = {
            key: list(encoded[key][feature_index])
            for key in tokenizer.model_input_names
            if key in encoded and key != "offset_mapping"
        }
        row["input_ids"] = input_ids
        row["start_positions"], row["end_positions"] = aligned_span or (cls_position, cls_position)
        row["overflow_to_sample_mapping"] = source_index
        features.append(row)
    return features


def prepare_validation_features(tokenizer: Any, examples: Sequence[dict[str, Any]], max_length: int, stride: int) -> list[dict[str, Any]]:
    """Create validation windows with enough metadata to restore spans in original contexts."""
    if not examples:
        return []
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for offset and sequence alignment")

    questions = [str(example["question"]).lstrip() for example in examples]
    contexts = [str(example["context"]) for example in examples]
    encoded = tokenizer(
        questions, contexts, truncation="only_second", max_length=max_length,
        stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
        padding=False,
    )
    mapping = encoded.get("overflow_to_sample_mapping")
    if mapping is None:
        raise KeyError("Tokenizer did not return overflow_to_sample_mapping")

    features: list[dict[str, Any]] = []
    for feature_index, source_index_value in enumerate(mapping):
        source_index = int(source_index_value)
        if not 0 <= source_index < len(examples):
            raise ValueError(f"Invalid overflow source index: {source_index}")
        sequence_ids = encoded.sequence_ids(feature_index)
        raw_offsets = encoded["offset_mapping"][feature_index]
        offsets = []
        for offset, sequence_id in zip(raw_offsets, sequence_ids):
            if sequence_id != 1 or offset is None or len(offset) != 2 or int(offset[1]) <= int(offset[0]):
                offsets.append(None)
            else:
                offsets.append((int(offset[0]), int(offset[1])))

        row = {
            key: list(encoded[key][feature_index])
            for key in tokenizer.model_input_names
            if key in encoded and key != "offset_mapping"
        }
        row.update({
            "example_id": str(examples[source_index].get("id", source_index)),
            "offset_mapping": offsets,
            "sequence_ids": list(sequence_ids),
            "overflow_to_sample_mapping": source_index,
        })
        features.append(row)
    return features


def window_answer_coverage(example: dict[str, Any], feature: dict[str, Any]) -> bool:
    """Return whether this feature contains at least one full annotated answer span."""
    answers = example.get("answers", {})
    texts = list(answers.get("text") or [])
    starts = list(answers.get("answer_start") or [])
    offsets = feature.get("offset_mapping")
    sequence_ids = feature.get("sequence_ids")
    if offsets is None or sequence_ids is None:
        raise KeyError("Validation feature must preserve offsets and sequence_ids")
    return any(
        character_to_token_span(offsets, sequence_ids, int(start), str(text)) is not None
        for text, start in zip(texts, starts)
    )


def decode_qa_predictions(examples: Sequence[dict[str, Any]], features: Sequence[dict[str, Any]], start_logits: Any, end_logits: Any, max_answer_length: int = 30) -> list[dict[str, Any]]:
    """Decode the best valid context span per example, aggregating all overflow windows."""
    if max_answer_length < 1:
        raise ValueError("max_answer_length must be positive")

    def rows(values: Any) -> list[list[float]]:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if hasattr(values, "tolist"):
            values = values.tolist()
        values = list(values)
        if values and isinstance(values[0], (int, float)):
            return [[float(value) for value in values]]
        return [[float(value) for value in (row.tolist() if hasattr(row, "tolist") else row)] for row in values]

    starts_by_feature, ends_by_feature = rows(start_logits), rows(end_logits)
    if len(starts_by_feature) != len(features) or len(ends_by_feature) != len(features):
        raise ValueError("There must be one start/end logit row for each feature")

    id_to_index = {str(example.get("id", index)): index for index, example in enumerate(examples)}
    best: dict[int, tuple[float, int, int, int, int, int]] = {}
    # tuple: score, feature index, start token, end token, start char, end char
    for feature_index, feature in enumerate(features):
        if "overflow_to_sample_mapping" in feature:
            example_index = int(feature["overflow_to_sample_mapping"])
        else:
            example_id = str(feature.get("example_id", ""))
            if example_id not in id_to_index:
                raise KeyError(f"Feature has no valid source-example mapping: {example_id!r}")
            example_index = id_to_index[example_id]
        if not 0 <= example_index < len(examples):
            raise ValueError(f"Invalid feature source index: {example_index}")

        offsets = feature.get("offset_mapping")
        sequence_ids = feature.get("sequence_ids")
        if offsets is None or sequence_ids is None or len(offsets) != len(sequence_ids):
            raise ValueError("Each validation feature needs aligned offsets and sequence_ids")
        start_row, end_row = starts_by_feature[feature_index], ends_by_feature[feature_index]
        usable = []
        for token_index, (offset, sequence_id) in enumerate(zip(offsets, sequence_ids)):
            if token_index >= len(start_row) or token_index >= len(end_row) or sequence_id != 1 or offset is None:
                continue
            if len(offset) != 2 or int(offset[0]) < 0 or int(offset[1]) <= int(offset[0]):
                continue
            usable.append(token_index)
        if not usable:
            continue

        # The top-20 context starts/ends follow the standard QA n-best candidate strategy.
        start_candidates = sorted(usable, key=lambda i: (-start_row[i], i))[:20]
        end_candidates = sorted(usable, key=lambda i: (-end_row[i], i))[:20]
        context = str(examples[example_index]["context"])
        for start_index in start_candidates:
            for end_index in end_candidates:
                token_length = end_index - start_index + 1
                if end_index < start_index or token_length > max_answer_length:
                    continue
                start_char = int(offsets[start_index][0])
                end_char = int(offsets[end_index][1])
                if start_char < 0 or end_char <= start_char or end_char > len(context):
                    continue
                score = start_row[start_index] + end_row[end_index]
                candidate = (score, feature_index, start_index, end_index, start_char, end_char)
                current = best.get(example_index)
                # Highest score wins; ties prefer earlier feature and token positions.
                if current is None or (score, -feature_index, -start_index, -end_index) > (
                    current[0], -current[1], -current[2], -current[3]
                ):
                    best[example_index] = candidate

    predictions = []
    for example_index, example in enumerate(examples):
        candidate = best.get(example_index)
        if candidate is None:
            prediction, start_char, end_char = "", None, None
        else:
            _, _, _, _, start_char, end_char = candidate
            prediction = str(example["context"])[start_char:end_char]
        predictions.append({
            "id": str(example.get("id", example_index)), "prediction": prediction,
            "start_char": start_char, "end_char": end_char,
        })
    return predictions


def normalize_answer(text: str) -> str:
    """Apply the official SQuAD lowercase/punctuation/article/whitespace normalization."""
    import string

    text = str(text or "").lower()
    punctuation = set(string.punctuation)
    text = "".join(char for char in text if char not in punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, gold_answers: Sequence[str]) -> float:
    """Return the best normalized Exact Match over references on a 0–1 scale."""
    references = list(gold_answers or [""])
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(reference) for reference in references))


def token_f1_score(prediction: str, gold_answers: Sequence[str]) -> float:
    """Return maximum multiset token-overlap F1 over references on a 0–1 scale."""
    from collections import Counter

    prediction_tokens = normalize_answer(prediction).split()
    references = list(gold_answers or [""])
    scores = []
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        if not prediction_tokens or not reference_tokens:
            scores.append(float(prediction_tokens == reference_tokens))
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def score_one_example(prediction: str, gold_answers: Sequence[str]) -> dict[str, float]:
    """Return per-example SQuAD EM/F1 on the official 0–100 scale."""
    references = list(gold_answers or [""])
    return {
        "em": 100.0 * exact_match_score(prediction, references),
        "f1": 100.0 * token_f1_score(prediction, references),
    }


def evaluate_predictions(predictions: Sequence[dict[str, Any]], references: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Aggregate EM/F1 by reference ID; missing predictions count as empty strings."""
    prediction_by_id: dict[str, str] = {}
    for row in predictions:
        sample_id = str(row["id"])
        if sample_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction ID: {sample_id}")
        prediction_by_id[sample_id] = str(row.get("prediction", row.get("answer", "")) or "")

    reference_by_id: dict[str, Sequence[str]] = {}
    for row in references:
        sample_id = str(row["id"])
        if sample_id in reference_by_id:
            raise ValueError(f"Duplicate reference ID: {sample_id}")
        answers = row.get("answers", {})
        if isinstance(answers, dict):
            gold_answers = answers.get("text", [])
        else:
            gold_answers = answers or []
        reference_by_id[sample_id] = [str(answer) for answer in gold_answers]

    if not reference_by_id:
        return {"em": 0.0, "f1": 0.0}
    scores = [
        score_one_example(prediction_by_id.get(sample_id, ""), gold_answers)
        for sample_id, gold_answers in reference_by_id.items()
    ]
    count = len(scores)
    return {
        "em": sum(score["em"] for score in scores) / count,
        "f1": sum(score["f1"] for score in scores) / count,
    }


def classify_span_errors(question: str, context: str, gold_answers: dict[str, Sequence[Any]], prediction: dict[str, Any]) -> dict[str, Any]:
    """Classify predicted span boundaries against all gold character spans."""
    del question  # Error type describes observed span geometry, not semantic cause.
    gold_texts = list(gold_answers.get("text", []))
    gold_starts = list(gold_answers.get("answer_start", []))
    gold_spans = []
    for text, start in zip(gold_texts, gold_starts):
        text, start = str(text), int(start)
        end = start + len(text)
        if text and 0 <= start < end <= len(context) and context[start:end] == text:
            gold_spans.append((start, end))

    pred_start = prediction.get("start_char")
    pred_end = prediction.get("end_char")
    if pred_start is not None and pred_end is not None:
        pred_start, pred_end = int(pred_start), int(pred_end)
        if pred_start < 0 or pred_end <= pred_start or pred_end > len(context):
            pred_start = pred_end = None
    else:
        pred_start = pred_end = None

    if not gold_spans:
        return {"wrong_start": True, "wrong_end": True, "wrong_span": True,
                "error_type": "unscorable", "overlap_type": "none"}

    exact = pred_start is not None and (pred_start, pred_end) in gold_spans
    if exact:
        return {"wrong_start": False, "wrong_end": False, "wrong_span": False,
                "error_type": "correct", "overlap_type": "full"}

    if pred_start is None:
        selected_gold = gold_spans[0]
        overlap = 0
    else:
        def span_evidence(span: tuple[int, int]):
            gold_start, gold_end = span
            overlap_chars = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
            matched_boundaries = int(pred_start == gold_start) + int(pred_end == gold_end)
            boundary_distance = abs(pred_start - gold_start) + abs(pred_end - gold_end)
            return matched_boundaries, overlap_chars, -boundary_distance
        selected_gold = max(gold_spans, key=span_evidence)
        overlap = max(0, min(pred_end, selected_gold[1]) - max(pred_start, selected_gold[0]))

    wrong_start = pred_start != selected_gold[0]
    wrong_end = pred_end != selected_gold[1]
    overlap_type = "partial" if overlap else "none"
    if not overlap:
        error_type = "non_overlapping"
    elif wrong_start and wrong_end:
        error_type = "both_boundaries_wrong"
    elif wrong_start:
        error_type = "wrong_start_only"
    else:
        error_type = "wrong_end_only"
    return {"wrong_start": wrong_start, "wrong_end": wrong_end, "wrong_span": True,
            "error_type": error_type, "overlap_type": overlap_type}


def question_content_words(question: str) -> set[str]:
    """Return simple lowercase content-like tokens for the keyword baseline."""
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "what", "when", "where", "who", "whom", "which", "why", "how", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or"}
    return {token for token in re.findall(r"[\w']+", question.lower()) if token not in stopwords}


def split_context_sentences(context: str) -> list[tuple[str, int, int]]:
    """Split context into sentence candidates while retaining original char offsets."""
    rows: list[tuple[str, int, int]] = []
    cursor = 0
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", context):
        start, end = match.span()
        while start < end and context[start].isspace():
            start += 1
        while end > start and context[end - 1].isspace():
            end -= 1
        if start < end:
            rows.append((context[start:end], start, end))
        cursor = match.end()
    if not rows and context:
        rows.append((context, 0, len(context)))
    return rows


def keyword_overlap_baseline(question: str, context: str) -> dict[str, Any]:
    """Rank context sentences by question-keyword overlap; never reads gold answers."""
    candidates = split_context_sentences(context)
    if not candidates:
        return {"answer": "", "start_char": 0, "end_char": 0, "overlap_score": 0.0}
    keywords = question_content_words(question)
    scored = []
    for sentence, start, end in candidates:
        sentence_words = set(re.findall(r"[\w']+", sentence.lower()))
        overlap = len(keywords & sentence_words)
        score = overlap / max(len(keywords), 1)
        scored.append((score, -start, sentence, start, end))
    score, _, answer, start, end = max(scored)
    return {"answer": answer, "start_char": start, "end_char": end, "overlap_score": score}

