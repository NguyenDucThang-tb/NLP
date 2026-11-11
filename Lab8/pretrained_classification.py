"""lab 08: các hàm dùng chung cho tokenizer, classifier và thí nghiệm có kiểm soát."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

ROOT = Path(__file__).resolve().parent
LABELS = {0: "Negative", 1: "Positive"}
VERSION = "lab08-v2-parquet-loaders"
# IMDb repo có các file Parquet có nhãn; dùng chúng để không phụ thuộc imdb.py cũ.
IMDB_PARQUET_FILES = {
    "train": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text/test-00000-of-00001.parquet",
}
# Bản Parquet này được công bố như bản sao của fancyzhx/yelp_polarity.
YELP_TEST_PARQUET = "https://huggingface.co/datasets/ajay232/yelp_polarity/resolve/946245561baec308f5c7d9edcb4e35876ae8d7d3/plain_text/test-00000-of-00001.parquet"
RESULT_COLUMNS = [
    "experiment_id", "experiment_name", "run_id", "run_mode", "model_name",
    "initialization", "train_mode", "seed", "train_fraction", "train_samples",
    "validation_samples", "test_samples", "max_length", "batch_size", "learning_rate",
    "epochs", "split", "domain", "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "f1_binary", "training_time_seconds", "peak_gpu_memory_mb", "total_parameters",
    "trainable_parameters", "run_status", "best_epoch", "revision", "device",
    "gradient_accumulation_steps", "config_hash", "accuracy_delta_b_minus_a",
    "f1_macro_delta_b_minus_a", "encoder_trainable",
]


@dataclass
class Config:
    mode: str = "quick"
    model_name: str = "distilbert/distilbert-base-uncased"
    revision: str = "main"
    seed: int = 42
    validation_fraction: float = 0.2
    train_size: int | None = None
    validation_size: int | None = None
    test_size: int | None = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int | None = None
    max_length: int = 128
    fp16: bool = True
    device: str = "auto"
    cpu_threads: int = 4
    high_confidence: float = 0.90
    medium_confidence: float = 0.70
    output_dir: str = str(ROOT)
    imdb_dir: str | None = None
    domain_b_file: str | None = None
    local_files_only: bool = False
    scratch_data_curve: bool = False

    def __post_init__(self):
        if self.mode not in {"quick", "full", "smoke"}:
            raise ValueError("mode phải là quick, full hoặc smoke")
        defaults = {"quick": (2000, 500, 500, 2), "smoke": (32, 16, 16, 1),
                    "full": (None, None, None, 3)}[self.mode]
        if self.mode != "full":
            for field, value in zip(("train_size", "validation_size", "test_size"), defaults[:3]):
                if getattr(self, field) is None:
                    setattr(self, field, value)
        if self.epochs is None:
            self.epochs = defaults[3]
        if not 0 < self.medium_confidence < self.high_confidence <= 1:
            raise ValueError("ngưỡng confidence không hợp lệ")
        if self.max_length not in range(4, 513):
            raise ValueError("max_length phải trong [4, 512]")
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, batch_size và accumulation phải dương")
        if self.mode == "full" and any(v is not None for v in (
            self.train_size, self.validation_size, self.test_size
        )):
            raise ValueError("full mode dùng toàn bộ dữ liệu; hãy dùng quick cho tập con")


def set_seed(seed=42):
    # cố định seed trước mỗi model để các run độc lập có thể lặp lại.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment_info():
    versions = {}
    for package in ["torch", "transformers", "datasets", "accelerate", "scikit-learn",
                    "numpy", "pandas", "matplotlib", "nbformat", "nbclient"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), **versions,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor() or platform.machine()}


def load_tokenizer(model_name="distilbert/distilbert-base-uncased", revision="main",
                   local_files_only=False):
    return AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True,
                                         local_files_only=local_files_only)


def validate_text_frame(frame, name):
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{name} phải có text và label")
    if frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"{name} có missing text/label; cần kiểm tra dữ liệu gốc")
    if not frame["text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{name} có text rỗng hoặc sai kiểu")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{name} có label ngoài 0=Negative, 1=Positive")
    if set(frame["label"]) != {0, 1}:
        raise ValueError(f"{name} phải có cả hai lớp")


def load_imdb_dataset(data_dir=None):
    if data_dir:
        data = {s: pd.read_csv(Path(data_dir) / f"{s}.csv") for s in ("train", "test")}
    else:
        # chỉ đọc hai split có nhãn; không gộp unsupervised.
        raw = load_dataset("parquet", data_files=IMDB_PARQUET_FILES)
        data = {s: raw[s].to_pandas()[["text", "label"]] for s in ("train", "test")}
    for split, frame in data.items():
        validate_text_frame(frame, f"IMDb {split}")
        frame["source_id"] = [f"imdb-{split}-{i}" for i in range(len(frame))]
    return data


def sample_stratified(frame, size, seed=42):
    if size is None or size >= len(frame):
        return frame.copy().reset_index(drop=True)
    if size < 4:
        raise ValueError("tập con cần ít nhất 4 mẫu")
    selected, _ = train_test_split(frame, train_size=size, random_state=seed,
                                  stratify=frame["label"])
    return selected.sort_values("source_id").reset_index(drop=True)


def create_stratified_splits(data, config):
    train, valid = train_test_split(data["train"], test_size=config.validation_fraction,
                                    random_state=config.seed, stratify=data["train"]["label"])
    frames = {"train": sample_stratified(train, config.train_size, config.seed),
              "validation": sample_stratified(valid, config.validation_size, config.seed),
              "test": sample_stratified(data["test"], config.test_size, config.seed)}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(frames[a].source_id) & set(frames[b].source_id):
            raise AssertionError(f"source_id overlap: {a}/{b}")
    return frames


def audit_splits(frames):
    rows = []
    for split, f in frames.items():
        rows.append({"split": split, "samples": len(f), "negative": int((f.label == 0).sum()),
                     "positive": int((f.label == 1).sum()),
                     "duplicate_extra_rows": int(f.text.duplicated().sum())})
    overlaps = []
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        common = set(frames[a].text) & set(frames[b].text)
        overlaps.append({"split_a": a, "split_b": b, "shared_exact_texts": len(common)})
        if common:
            logging.warning("%s/%s có %s văn bản trùng; giữ split chính thức và báo nguy cơ leakage", a, b, len(common))
    return pd.DataFrame(rows), pd.DataFrame(overlaps)


def create_training_subset(train, fraction, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError("fraction phải trong (0,1]")
    # cùng thứ tự ngẫu nhiên trong mỗi lớp giúp 10% nằm trong 25%, 50%, 100%.
    selected = []
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        group = train[train.label == label]
        order = rng.permutation(len(group))
        n = max(1, int(round(len(group) * fraction)))
        selected.append(group.iloc[order[:n]])
    return pd.concat(selected).sort_values("source_id").reset_index(drop=True)


def freeze_encoder(model):
    for p in model.base_model.parameters():
        p.requires_grad = False
    encoder_ids = {id(p) for p in model.base_model.parameters()}
    head = [p for p in model.parameters() if id(p) not in encoder_ids]
    if not head or not all(p.requires_grad for p in head):
        raise AssertionError("classification head phải còn trainable")
    assert not any(p.requires_grad for p in model.base_model.parameters())
    return model


def count_parameters(model):
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_trainable": any(p.requires_grad for p in model.base_model.parameters())}


def build_model(config, initialization="pretrained", train_mode="fine_tuning"):
    set_seed(config.seed)
    architecture = AutoConfig.from_pretrained(config.model_name, revision=config.revision,
                                               local_files_only=config.local_files_only)
    architecture.num_labels = 2
    architecture.id2label = LABELS
    architecture.label2id = {v: k for k, v in LABELS.items()}
    if initialization == "scratch":
        # from_config chỉ tạo trọng số ngẫu nhiên; không tải pretrained weights.
        model = AutoModelForSequenceClassification.from_config(architecture)
    elif initialization == "pretrained":
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, revision=config.revision, config=architecture,
            local_files_only=config.local_files_only)
    else:
        raise ValueError("initialization phải là pretrained hoặc scratch")
    if train_mode == "frozen":
        freeze_encoder(model)
    elif train_mode != "fine_tuning":
        raise ValueError("train_mode không hợp lệ")
    else:
        assert all(p.requires_grad for p in model.base_model.parameters())
    return model


class EncodedReviews(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.encoding = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length,
                                  padding=False)
        self.labels = frame.label.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.encoding.items()}, "labels": self.labels[i]}


def make_loader(frame, tokenizer, config, max_length, shuffle=False):
    collator = DataCollatorWithPadding(tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(EncodedReviews(frame, tokenizer, max_length),
                      batch_size=config.batch_size, shuffle=shuffle, collate_fn=collator,
                      generator=generator, num_workers=0)


def compute_metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
            "f1_binary": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))}


def predict_with_confidence(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.inference_mode():
        for batch in loader:
            labels_all.extend(batch.pop("labels").tolist())
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits_all.append(model(**inputs).logits.float().cpu())
    logits = torch.cat(logits_all)
    probabilities = torch.softmax(logits, dim=-1).numpy()
    if probabilities.shape != (len(labels_all), 2) or not np.isfinite(probabilities).all():
        raise AssertionError("probability shape/giá trị không hợp lệ")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise AssertionError("tổng probability không bằng 1")
    return np.asarray(labels_all), probabilities.argmax(1), probabilities, logits.numpy()


def evaluate_model(model, loader, device):
    labels, pred, proba, logits = predict_with_confidence(model, loader, device)
    loss = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    return {**compute_metrics(labels, pred), "loss": loss}, (labels, pred, proba)


def train_model(model, train_loader, val_loader, config, device, run_dir, frozen=False):
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    amp = device.type == "cuda" and config.fp16
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    history, best_f1, best_epoch = [], -1.0, 0
    training_time, validation_time = 0.0, 0.0
    checkpoint = run_dir / "best_model"
    for epoch in range(1, config.epochs + 1):
        model.train()
        if frozen:
            # tắt dropout trong encoder frozen để features ổn định; head vẫn ở train mode.
            model.base_model.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0.0, 0
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # nhóm cuối có thể thiếu micro-batch, nên chia loss theo số batch thật của nhóm.
            group_start = step // config.gradient_accumulation_steps * config.gradient_accumulation_steps
            group_size = min(config.gradient_accumulation_steps, len(train_loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss không hữu hạn")
            n = batch["labels"].shape[0]
            loss_sum += loss.item() * n
            seen += n
            scaler.scale(loss / group_size).backward()
            if (step + 1) % config.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda": torch.cuda.synchronize()
        training_time += time.perf_counter() - start
        start = time.perf_counter()
        metrics, _ = evaluate_model(model, val_loader, device)
        if device.type == "cuda": torch.cuda.synchronize()
        validation_time += time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                        "validation_loss": metrics["loss"], "validation_f1_macro": metrics["f1_macro"]})
        logging.info("epoch=%s train_loss=%.5f val_loss=%.5f val_macro_f1=%.5f", epoch,
                     loss_sum / seen, metrics["loss"], metrics["f1_macro"])
        if metrics["f1_macro"] > best_f1:
            best_f1, best_epoch = metrics["f1_macro"], epoch
            # chọn checkpoint bằng validation macro-F1; không xem test khi chọn.
            model.save_pretrained(checkpoint)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None
    del optimizer, scaler, model
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    best = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    if frozen: freeze_encoder(best)
    best.eval()
    metrics, _ = evaluate_model(best, val_loader, device)
    return best, metrics, {"training_time_seconds": training_time,
        "validation_time_seconds": validation_time, "peak_gpu_memory_mb": peak,
        "cpu_peak_memory_mb": None, "memory_note": "CPU peak not measured; GPU peak includes model/training/validation when available",
        "best_epoch": best_epoch}


def save_experiment_result(path, row):
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        if not np.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
            raise ValueError(f"không lưu metric không hợp lệ: {metric}")
    path = Path(path)
    frame = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame(columns=RESULT_COLUMNS)
    for col in RESULT_COLUMNS:
        if col not in frame: frame[col] = None
    key = ["experiment_id", "run_id", "split", "domain"]
    if not frame.empty:
        duplicate = np.ones(len(frame), dtype=bool)
        for col in key:
            duplicate &= frame[col].astype(str).eq(str(row[col])).to_numpy()
        frame = frame.loc[~duplicate]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def plot_data_efficiency(results, path, mode):
    data = results[(results.experiment_id.astype(str) == "16") & (results.split == "validation") & (results.run_mode == mode)]
    if data.empty: return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in data.groupby("initialization"):
        group = group.sort_values("train_fraction")
        ax.plot(group.train_fraction * 100, group.f1_macro, "o-", label=name)
    ax.set(xlabel="Training data (%)", ylabel="Validation macro-F1", xticks=[10, 25, 50, 100],
           title=f"Data efficiency — {mode} (measured points only)", ylim=(0, 1))
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_confusion_matrix(labels, pred, path, title):
    matrix = confusion_matrix(labels, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=list(LABELS.values()),
           yticklabels=list(LABELS.values()), xlabel="Predicted", ylabel="Actual", title=title)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return matrix


def plot_confidence_analysis(labels, pred, proba, path, config):
    confidence = proba.max(axis=1)
    groups = np.where(confidence >= config.high_confidence, "High",
                      np.where(confidence >= config.medium_confidence, "Medium", "Low"))
    rows = []
    for group in ["High", "Medium", "Low"]:
        mask = groups == group
        count = int(mask.sum())
        correct = int(((labels == pred) & mask).sum())
        rows.append({"group": group, "count": count, "correct": correct, "incorrect": count - correct,
                     "accuracy": correct / count if count else None,
                     "mean_confidence": float(confidence[mask].mean()) if count else None})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.group, table.correct, label="Correct")
    ax.bar(table.group, table.incorrect, bottom=table.correct, label="Incorrect")
    ax.set(xlabel=f"Confidence: High ≥ {config.high_confidence}; Medium ≥ {config.medium_confidence}",
           ylabel="Predictions", title=f"Confidence and correctness — {config.mode}")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return table


def extract_error_examples(frame, labels, pred, proba, config):
    out = frame[["source_id", "text"]].copy().reset_index(drop=True)
    out["true_label"], out["predicted_label"] = labels, pred
    out["p_negative"], out["p_positive"] = proba[:, 0], proba[:, 1]
    out["confidence"] = proba.max(axis=1)
    out["correct"] = labels == pred
    out["outcome"] = np.where(labels == pred, "correct", np.where(pred == 1, "FP", "FN"))
    errors = out[~out.correct].sort_values(["confidence", "source_id"], ascending=[False, True])
    selections, availability = [], {}
    for group, candidates, requested in [
        ("FP", errors[errors.outcome == "FP"], 3),
        ("FN", errors[errors.outcome == "FN"], 3),
        ("high_confidence_wrong", errors[errors.confidence >= config.high_confidence], 2),
    ]:
        availability[group] = {"available": len(candidates), "requested": requested,
                               "missing": max(0, requested - len(candidates))}
        selections.extend(candidates.head(requested).source_id.tolist())
    chosen = errors[errors.source_id.isin(selections)].copy()
    chosen["error_type"] = "TODO(student)"
    chosen["hypothesis"] = "TODO(student)"
    return out, chosen, availability


class LabRunner:
    def __init__(self, config: Config):
        self.config = config
        self.root = Path(config.output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output = self.root / "outputs" / config.mode
        self.output.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(self.output / "run.log")])
        self.device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                                   else "cpu" if config.device == "auto" else config.device)
        if self.device.type == "cpu": torch.set_num_threads(config.cpu_threads)
        set_seed(config.seed)
        self.tokenizer = load_tokenizer(config.model_name, config.revision, config.local_files_only)
        self.frames = create_stratified_splits(load_imdb_dataset(config.imdb_dir), config)
        overview, overlap = audit_splits(self.frames)
        overview.to_csv(self.output / "split_summary.csv", index=False)
        overlap.to_csv(self.output / "duplicate_text_audit.csv", index=False)
        for split, frame in self.frames.items():
            frame[["source_id", "label"]].to_csv(self.output / f"{split}_manifest.csv", index=False)
        self.fingerprint = hashlib.sha256(b"".join(
            pd.util.hash_pandas_object(f[["source_id", "text", "label"]], index=False).values.tobytes()
            for f in self.frames.values()
        )).hexdigest()
        self.runs = {}
        self.environment = {**environment_info(), "device": str(self.device),
                            "torch_cpu_threads": torch.get_num_threads()}
        (self.output / "environment.json").write_text(json.dumps(self.environment, indent=2))
        (self.output / "config.json").write_text(json.dumps(asdict(config), indent=2))
        if not (self.root / "results.csv").exists():
            pd.DataFrame(columns=RESULT_COLUMNS).to_csv(self.root / "results.csv", index=False)

    def status(self, experiment, state, message):
        path = self.output / "status.csv"
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["experiment", "status", "message"])
        frame = frame[frame.experiment.astype(str) != str(experiment)]
        frame = pd.concat([frame, pd.DataFrame([{"experiment": str(experiment), "status": state, "message": message}])], ignore_index=True)
        frame.to_csv(path, index=False)

    def run_model(self, initialization="pretrained", train_mode="fine_tuning", fraction=1.0,
                  max_length=None):
        c = self.config
        length = max_length or c.max_length
        subset = create_training_subset(self.frames["train"], fraction, c.seed)
        identity = {**asdict(c), "initialization": initialization, "train_mode": train_mode,
                    "train_fraction": fraction, "max_length": length,
                    "data_fingerprint": self.fingerprint, "implementation": VERSION,
                    "environment": self.environment}
        identity.pop("output_dir", None)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        run_id = f"{c.mode}-{initialization}-{train_mode}-f{fraction:g}-l{length}-{digest}"
        directory = self.output / run_id
        directory.mkdir(parents=True, exist_ok=True)
        complete = directory / "complete.json"
        if complete.exists() and (directory / "best_model" / "config.json").exists():
            logging.info("reuse completed run %s", run_id)
            record = json.loads(complete.read_text())
            if not (directory / "best_model" / "tokenizer_config.json").exists():
                self.tokenizer.save_pretrained(directory / "best_model")
            self.runs[run_id] = record
            return record
        (directory / "config.json").write_text(json.dumps(identity, indent=2))
        subset[["source_id", "label"]].to_csv(directory / "train_manifest.csv", index=False)
        logging.info("train %s samples=%s length=%s", run_id, len(subset), length)
        model = build_model(c, initialization, train_mode)
        params = count_parameters(model)
        revision = getattr(model.config, "_commit_hash", None) or c.revision
        started = time.perf_counter()
        best, metrics, timing = train_model(model,
            make_loader(subset, self.tokenizer, c, length, True),
            make_loader(self.frames["validation"], self.tokenizer, c, length),
            c, self.device, directory, train_mode == "frozen")
        record = {"run_id": run_id, "run_mode": c.mode, "model_name": c.model_name,
            "initialization": initialization, "train_mode": train_mode, "seed": c.seed,
            "train_fraction": fraction, "train_samples": len(subset),
            "validation_samples": len(self.frames["validation"]), "test_samples": len(self.frames["test"]),
            "max_length": length, "batch_size": c.batch_size, "learning_rate": c.learning_rate,
            "epochs": c.epochs, **params, **timing, "revision": revision, "device": str(self.device),
            "gradient_accumulation_steps": c.gradient_accumulation_steps,
            "config_hash": digest, "wall_time_seconds": time.perf_counter() - started,
            "validation_metrics": metrics, "checkpoint": str(directory / "best_model")}
        self.tokenizer.save_pretrained(directory / "best_model")
        complete.write_text(json.dumps(record, indent=2))
        del best, model
        gc.collect()
        if self.device.type == "cuda": torch.cuda.empty_cache()
        self.runs[run_id] = record
        return record

    def record(self, run, experiment, name, split="validation", domain="IMDb", metrics=None, **extra):
        metrics = run["validation_metrics"] if metrics is None else metrics
        row = {k: run.get(k) for k in RESULT_COLUMNS}
        row.update(experiment_id=str(experiment), experiment_name=name, split=split,
                   domain=domain, run_status="EXECUTED", **{k: v for k, v in metrics.items() if k in RESULT_COLUMNS}, **extra)
        save_experiment_result(self.root / "results.csv", row)
        return row

    def pretrained_vs_scratch(self):
        rows = []
        for init in ["scratch", "pretrained"]:
            rows.append(self.record(self.run_model(init), "14", "Pretrained vs scratch"))
        self.status("14", "EXECUTED", "same architecture/tokenizer, validation comparison")
        return pd.DataFrame(rows)

    def frozen_vs_finetuning(self):
        rows = [self.record(self.run_model("pretrained", mode), "15", "Frozen vs fine-tuning")
                for mode in ["frozen", "fine_tuning"]]
        self.status("15", "EXECUTED", "encoder requires_grad checked at model construction")
        return pd.DataFrame(rows)

    def data_efficiency(self):
        rows = []
        for init in (["pretrained", "scratch"] if self.config.scratch_data_curve else ["pretrained"]):
            for fraction in [.1, .25, .5, 1.0]:
                rows.append(self.record(self.run_model(init, fraction=fraction), "16", "Data efficiency"))
        # chỉ vẽ các run của lần gọi này để không trộn hai cấu hình quick khác nhau.
        plot_data_efficiency(pd.DataFrame(rows), self.root / "data_efficiency.png", self.config.mode)
        self.status("16", "EXECUTED", "all four nested fractions; scratch curve is optional")
        return pd.DataFrame(rows)

    def sequence_length(self):
        rows = [self.record(self.run_model(max_length=length), "17", "Sequence length")
                for length in [64, 128, 256]]
        self.status("17", "EXECUTED", "CPU peak memory not measured if no CUDA")
        return pd.DataFrame(rows)

    def load_run(self, run):
        # dựng lại đường dẫn từ run_id để có thể chuyển outputs sang Colab/Drive.
        checkpoint = self.output / run["run_id"] / "best_model"
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device)
        model.eval()
        return model

    def application(self):
        # chọn cấu hình tham chiếu từ trước, không chọn model bằng test metrics.
        run = self.run_model("pretrained", "fine_tuning", 1.0, self.config.max_length)
        model = self.load_run(run)
        metrics, (labels, pred, proba) = evaluate_model(model,
            make_loader(self.frames["test"], self.tokenizer, self.config, run["max_length"]), self.device)
        row = self.record(run, "19", "Sentiment application", "test", metrics=metrics)
        matrix = plot_confusion_matrix(labels, pred, self.root / "confusion_matrix.png",
                                        f"IMDb test — {self.config.mode} — {len(labels)} samples")
        report = classification_report(labels, pred, labels=[0, 1], target_names=list(LABELS.values()),
                                       output_dict=True, zero_division=0)
        (self.output / "classification_report.json").write_text(json.dumps(report, indent=2))
        all_predictions, candidates, availability = extract_error_examples(self.frames["test"], labels, pred, proba, self.config)
        all_predictions.to_csv(self.output / "test_predictions.csv", index=False)
        candidates.to_csv(self.output / "error_candidates.csv", index=False)
        (self.output / "error_availability.json").write_text(json.dumps(availability, indent=2))
        confidence = plot_confidence_analysis(labels, pred, proba, self.root / "confidence_analysis.png", self.config)
        confidence.to_csv(self.output / "confidence_groups.csv", index=False)
        write_error_template(self.root, candidates, availability, self.config)
        self.status("19", "EXECUTED", f"test n={len(labels)}; reference={run['run_id']}")
        self.status("20", "GENERATED", json.dumps(availability))
        self.status("21", "GENERATED", "thresholds recorded in config; empty groups have null accuracy")
        del model
        gc.collect()
        return pd.DataFrame([row]), pd.DataFrame(matrix, index=LABELS.values(), columns=LABELS.values()), confidence

    def domain_shift(self):
        run = self.run_model("pretrained", "fine_tuning", 1.0, self.config.max_length)
        if self.config.domain_b_file:
            domain = pd.read_csv(self.config.domain_b_file)
            domain_name = "External sentiment: " + Path(self.config.domain_b_file).name
        else:
            try:
                # Dùng file Parquet test của bản sao công khai để tránh builder script cũ.
                domain = load_dataset("parquet", data_files={"test": YELP_TEST_PARQUET},
                                      split="test").to_pandas()[["text", "label"]]
            except Exception as error:
                raise OSError("Domain B chưa tải được; dùng cache hoặc domain_b_file với text,label thật") from error
            domain_name = "Yelp Polarity"
        validate_text_frame(domain, "Yelp/Domain B")
        domain["source_id"] = [f"domain-b-test-{i}" for i in range(len(domain))]
        domain = sample_stratified(domain, self.config.test_size, self.config.seed)
        domain[["source_id", "label"]].to_csv(self.output / "domain_b_manifest.csv", index=False)
        model = self.load_run(run)
        rows = []
        for name, frame in [("IMDb", self.frames["test"]), (domain_name, domain)]:
            metrics, _ = evaluate_model(model, make_loader(frame, self.tokenizer, self.config, run["max_length"]), self.device)
            rows.append(self.record(run, "18", "Domain shift", "test", name, metrics,
                                   **({"test_samples": len(frame)})))
        delta = {"accuracy_delta_b_minus_a": rows[1]["accuracy"] - rows[0]["accuracy"],
                 "f1_macro_delta_b_minus_a": rows[1]["f1_macro"] - rows[0]["f1_macro"]}
        rows[1].update(delta)
        save_experiment_result(self.root / "results.csv", rows[1])
        (self.output / "domain_label_mapping.json").write_text(json.dumps({"IMDb": LABELS, domain_name: LABELS,
            "domain_b_source": self.config.domain_b_file or "fancyzhx/yelp_polarity: test (Parquet mirror ajay232/yelp_polarity)",
            "yelp_raw_label_names": ["1", "2"] if not self.config.domain_b_file else None,
            "note": "compatible binary sentiment; Domain B is test only, never used for fitting or checkpoint selection"}, indent=2))
        self.status("18", "EXECUTED", f"A=IMDb B={domain_name}; delta=B-A; nB={len(domain)}")
        del model
        gc.collect()
        return pd.DataFrame(rows)

    def run_all(self, experiments=("14", "15", "16", "17", "18", "19")):
        methods = {"14": self.pretrained_vs_scratch, "15": self.frozen_vs_finetuning,
                   "16": self.data_efficiency, "17": self.sequence_length,
                   "18": self.domain_shift, "19": self.application}
        for exp in experiments:
            try:
                methods[exp]()
            except Exception as exc:
                self.status(exp, "BLOCKED" if isinstance(exc, (OSError, ConnectionError)) else "FAILED", str(exc))
                (self.output / f"exception_{exp}.txt").write_text(traceback.format_exc())
                logging.exception("experiment %s failed", exp)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.write_completion_report()
        return pd.read_csv(self.root / "results.csv")

    def write_completion_report(self):
        results = pd.read_csv(self.root / "results.csv")
        verify_path = self.root / "outputs" / "verification.json"
        verification = json.loads(verify_path.read_text()) if verify_path.exists() else {}
        status_path = self.output / "status.csv"
        statuses = pd.read_csv(status_path, dtype=str) if status_path.exists() else pd.DataFrame()
        def state(exp):
            if statuses.empty: return "NOT_RUN"
            s = statuses[statuses.experiment == str(exp)]
            return s.iloc[-1].status if len(s) else "NOT_RUN"
        todo = {
            9: "STUDENT TODO: tự tính MLM loss", 10: "STUDENT TODO: ghi và giải thích tokens",
            11: "STUDENT TODO: prediction trước experiment", 12: "STUDENT TODO: giải thích mask",
            13: "GPU/T4 và BERT-base NOT_RUN", 20: "STUDENT TODO: error type và hypothesis",
            21: "STUDENT TODO: interpretation confidence", 23: "GPU reproducibility NOT_RUN",
            24: "Sinh viên tự rà soát khai báo AI", 27: "STUDENT TODO: 12 câu learning check",
            28: "STUDENT TODO: 6 câu observation/evidence/interpretation",
            29: "STUDENT TODO: reflection 300–500 từ",
        }
        rows = []
        for exp in [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,27,28,29]:
            file = "calculations.md" if exp in [9,27] else "prediction.md" if exp==11 else "tokenization.ipynb" if exp in [10,12] else "pretrained_classification.py" if exp==13 else "error_analysis.md" if exp in [20,28] else "reflection.md" if exp==29 else "README.md" if exp in [23,24] else "fine_tuning.ipynb"
            personal = exp in [9,11,27,28,29]
            if exp in [14,15,16,17,18,19,20,21]:
                executed = state(exp)
            elif personal:
                executed = "STUDENT TODO"
            elif exp in [10,12]:
                executed = verification.get("notebooks", {}).get("tokenization.ipynb", {}).get("status", "NOT_RUN")
            elif exp == 13:
                executed = verification.get("notebooks", {}).get("fine_tuning.ipynb", {}).get("status", "NOT_RUN")
            elif exp == 22:
                executed = "EXECUTED" if any(state(i) == "EXECUTED" for i in [14,15,16,17]) else "NOT_RUN"
            else:
                executed = "GENERATED"
            if exp in [14,15,16,17,18,19,20,21] and executed in {"EXECUTED", "GENERATED"}:
                executed += " (" + self.config.mode + ")"
            evidence = "outputs/" + self.config.mode + "/status.csv; results.csv"
            if personal: evidence = file
            elif exp in [10,12,13]: evidence = "saved notebook output; outputs/verification.json"
            elif exp == 20: evidence = "outputs/" + self.config.mode + "/error_candidates.csv"
            elif exp == 21: evidence = "outputs/" + self.config.mode + "/confidence_groups.csv"
            elif exp in [23,24]: evidence = "README.md; config.json; environment.json; verification.json"
            rows.append({"Bài":exp,"File":file,"Implemented":"template" if personal else "IMPLEMENTED",
                         "Executed":executed,"Evidence":evidence,
                         "Remaining work":todo.get(exp,"Full/GPU NOT_RUN; STUDENT TODO: interpretation")})
        text = "# Kiểm tra hoàn thành LAB 08\n\nMode thực tế: `"+self.config.mode+"`. Các experiment chưa có EXECUTED không được xem là đã chạy.\n\n"
        text += pd.DataFrame(rows).to_markdown(index=False)+"\n\n## Kết quả đã đo\n\n"
        subset = results[results.run_mode==self.config.mode]
        text += subset[["experiment_id","initialization","train_mode","train_fraction","train_samples","max_length","split","domain","accuracy","f1_macro","training_time_seconds"]].to_markdown(index=False) if len(subset) else "Chưa có metrics."
        text += "\n\nCPU peak memory: not measured nếu cột trống. Quick/smoke không thay cho full. Full experiments và GPU/T4 chưa chạy thì vẫn NOT_RUN, dù quick có EXECUTED. Interpretation, calculations, prediction, reflection và giả thuyết lỗi do sinh viên tự hoàn thiện.\n"
        (self.root / "completion_report.md").write_text(text)


def write_error_template(root, candidates, availability, config):
    text = f"# Bài 20 — Error analysis\n\nBằng chứng lấy từ IMDb test, mode `{config.mode}`, model pretrained fine-tuned, checkpoint chọn bằng validation macro-F1.\n\n"
    text += f"Model: `{config.model_name}`; max_length={config.max_length}; seed={config.seed}. Run ID/checkpoint chính xác nằm ở record Bài 19 trong results.csv và complete.json.\n\n"
    text += "## Số lượng mẫu có thật\n\n"+pd.DataFrame(availability).T.to_markdown()+"\n\n"
    text += "High-confidence wrong dùng ngưỡng `"+str(config.high_confidence)+"`. Không đủ mẫu ở nhóm nào được ghi ở cột missing; không tạo mẫu bù. Toàn bộ review/predictions: `outputs/"+config.mode+"/test_predictions.csv`.\n\n"
    for row in candidates.itertuples():
        text += f"### {row.source_id} — {row.outcome}\n\nText:\n\n> "+row.text.replace('\n','\n> ')+"\n\n"
        text += f"- True label: {LABELS[row.true_label]}\n- Prediction: {LABELS[row.predicted_label]}\n- Confidence: {row.confidence:.8f}\n- Error type: TODO(student)\n- Hypothesis: TODO(student)\n\n"
    text += "Các nhóm gợi ý để tự kiểm tra: negation, sarcasm, long-range dependency, ambiguous sentiment, domain-specific expression, rare word, tokenization, context, label ambiguity. Không xem keyword là bằng chứng nhân quả.\n\n## Bài 28 — Sáu câu hỏi bắt buộc\n\n"
    questions = ["Model nào hưởng lợi nhiều hơn khi training data ít?", "Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?", "Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?", "Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?", "Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?", "Nếu sequence length tăng nhưng performance không đổi trong khi training time tăng mạnh, có nên tăng tiếp không?"]
    for i, question in enumerate(questions, 1):
        text += f"### Q{i}. {question}\n\n- Observation: TODO(student)\n- Evidence: TODO(student): tham chiếu results.csv; chưa có đủ thực nghiệm thì ghi rõ.\n- Interpretation: TODO(student)\n\n"
    (root / "error_analysis.md").write_text(text)


def infer_texts(model, tokenizer, texts, device, max_length=128):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    model.eval()
    with torch.inference_mode():
        output = model(**{k:v.to(device) for k,v in inputs.items()}, output_hidden_states=True)
        proba = torch.softmax(output.logits.float(), dim=-1).cpu().numpy()
    return pd.DataFrame({"text":texts,"logits":output.logits.cpu().tolist(),
        "p_negative":proba[:,0],"p_positive":proba[:,1],
        "prediction":[LABELS[int(i)] for i in proba.argmax(1)]}), output.hidden_states[-1].cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick","full","smoke"], default="quick")
    parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--experiments", nargs="+", choices=["14","15","16","17","18","19"],
                        default=["14","15","16","17","18","19"])
    for name in ["train-size","validation-size","test-size","epochs","batch-size","max-length","cpu-threads","gradient-accumulation-steps"]:
        parser.add_argument("--"+name, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(ROOT))
    parser.add_argument("--imdb-dir")
    parser.add_argument("--domain-b-file")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--scratch-data-curve", action="store_true")
    parser.add_argument("--inference", nargs="+")
    parser.add_argument("--checkpoint")
    args=parser.parse_args()
    fields={k:v for k,v in vars(args).items() if k not in {"experiments","inference","checkpoint"} and v is not None}
    config=Config(**fields)
    if args.inference:
        if args.checkpoint:
            tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
            model=AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
        else:
            tokenizer=load_tokenizer(config.model_name, config.revision, config.local_files_only)
            model=build_model(config)
            print("WARNING: pretrained encoder with RANDOM task head; not an IMDb-trained classifier.")
        device=torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                            else "cpu" if config.device == "auto" else config.device)
        table,representation=infer_texts(model.to(device),tokenizer,args.inference,device,config.max_length)
        print(model)
        print(count_parameters(model))
        print(table.to_string(index=False)); print("contextual representation shape:",tuple(representation.shape))
    else:
        try:
            runner=LabRunner(config)
        except Exception:
            output=Path(config.output_dir)/"outputs"/config.mode
            output.mkdir(parents=True,exist_ok=True)
            (output/"initialization_exception.txt").write_text(traceback.format_exc())
            pd.DataFrame([{"experiment": e, "status": "BLOCKED",
                           "message": "initialization failed; see initialization_exception.txt"}
                          for e in args.experiments]).to_csv(output/"status.csv",index=False)
            raise
        stale_error = runner.output / "initialization_exception.txt"
        if stale_error.exists():
            # Giữ traceback cũ để đối chiếu, nhưng không để nhầm với lượt chạy mới.
            stale_error.replace(runner.output / "initialization_exception_previous.txt")
        print(runner.run_all(args.experiments).to_string(index=False))
        states=pd.read_csv(runner.output/"status.csv",dtype=str)
        if states[states.experiment.isin(args.experiments)].status.isin(["FAILED","BLOCKED"]).any():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
