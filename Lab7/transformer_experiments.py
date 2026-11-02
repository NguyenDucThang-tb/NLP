"""Shared IMDb pipeline for LAB 07 Transformer experiments.

The notebook is the entry point. This module keeps reusable data, model,
training and plotting code in one place so every experiment uses the same split.
"""

from __future__ import annotations

import copy
import html
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from positional_encoding import make_positional_encoding
from transformer_block import TransformerBlock


SEED = 42
SPLIT_SEED = 2026
D_MODEL = 128
D_FF = 512
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 5
N_BLOCKS = 1
MAX_LENGTH = 128
MAX_LENGTHS = (32, 64, 128, 256)
MAX_VOCAB_SIZE = 20_000
MIN_FREQUENCY = 2
PAD_ID = 0
UNK_ID = 1


def set_seed(seed: int = SEED) -> None:
    """Đặt seed cho Python, NumPy, PyTorch và CUDA nếu có."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    """Dùng cùng kiểu làm sạch/tokenizer regex với LAB 05."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


class ReviewDataset(Dataset):
    """Cắt sequence theo max_length; collate_fn sẽ thêm padding mỗi batch."""

    def __init__(self, sequences: list[list[int]], labels: np.ndarray, max_length: int):
        self.sequences = [sequence[:max_length] or [UNK_ID] for sequence in sequences]
        self.labels = torch.tensor(np.array(labels, copy=True), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


def collate_reviews(batch):
    """Pad bên phải tới độ dài lớn nhất trong batch hiện tại."""
    sequences, labels = zip(*batch)
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(batch), max_length), PAD_ID, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
    return input_ids, torch.stack(labels)


class TransformerClassifier(nn.Module):
    """Embedding + sinusoidal PE + TransformerBlock + masked pooling + classifier."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = 4,
        d_ff: int = D_FF,
        num_classes: int = 2,
        pooling: str = "mean",
        max_length: int = MAX_LENGTH,
        n_blocks: int = N_BLOCKS,
        dropout: float = 0.1,
    ):
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling phải là 'mean' hoặc 'cls'")
        if n_blocks < 1:
            raise ValueError("n_blocks phải ít nhất là 1")
        self.d_model = d_model
        self.pooling = pooling
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model)) if pooling == "cls" else None
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        pe_length = max_length + int(pooling == "cls")
        pe = torch.tensor(make_positional_encoding(pe_length, d_model), dtype=torch.float32)
        self.register_buffer("positional_encoding", pe.unsqueeze(0), persistent=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids phải có shape (B, T)")
        if attention_mask is None:
            attention_mask = input_ids.ne(PAD_ID)
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask phải cùng shape với input_ids")

        hidden = self.embedding(input_ids)
        if self.pooling == "cls":
            batch_size = input_ids.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1)
            hidden = torch.cat((cls, hidden), dim=1)
            cls_mask = torch.ones((batch_size, 1), device=input_ids.device, dtype=torch.bool)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        if hidden.shape[1] > self.positional_encoding.shape[1]:
            raise ValueError("sequence dài hơn positional encoding đã khởi tạo")
        positions = self.positional_encoding[:, :hidden.shape[1]].to(dtype=hidden.dtype)
        # Cộng PE chỉ ở token thật (và CLS nếu có); padding vẫn được mask.
        hidden = (hidden + positions) * attention_mask.unsqueeze(-1).to(hidden.dtype)

        last_attention = None
        for block in self.blocks:
            if return_attention:
                hidden, last_attention = block(
                    hidden, attention_mask=attention_mask, return_attention=True
                )
            else:
                hidden = block(hidden, attention_mask=attention_mask)

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden[:, 0]
        logits = self.classifier(pooled)
        if return_attention:
            return logits, last_attention
        return logits


class LSTMClassifier(nn.Module):
    """LSTM baseline dùng cùng vocabulary, embedding size và classifier."""

    def __init__(self, vocab_size: int, hidden_size: int = D_MODEL, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(D_MODEL, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids.ne(PAD_ID)
        lengths = attention_mask.sum(dim=1).clamp_min(1).to(torch.long)
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1]))


class ExperimentRunner:
    """Giữ chung split/dữ liệu và điều phối các bài 19–25, 29."""

    def __init__(
        self,
        output_dir: str | Path,
        device: str | torch.device | None = None,
        run_mode: str = "full",
    ):
        if run_mode not in {"full", "smoke"}:
            raise ValueError("run_mode phải là 'full' hoặc 'smoke'")
        self.run_mode = run_mode
        self.epochs = EPOCHS if run_mode == "full" else 1
        self.batch_size = BATCH_SIZE
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(min(10, torch.get_num_threads()))
        self.models: dict[str, nn.Module] = {}
        self.histories: dict[str, pd.DataFrame] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.times: dict[str, float] = {}
        self.records: list[dict] = []
        self._dataset_cache: dict[tuple[str, int], ReviewDataset] = {}
        self._loader_cache: dict[tuple[str, int, bool], DataLoader] = {}
        self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> None:
        """Dùng chính split LAB 05; test split gốc không bị chia lại."""
        imdb = load_dataset("imdb")
        train_df = imdb["train"].to_pandas()[["text", "label"]]
        test_df = imdb["test"].to_pandas()[["text", "label"]]
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.20,
            random_state=SPLIT_SEED,
            stratify=train_df["label"],
        )
        if self.run_mode == "smoke":
            train_df, _ = train_test_split(
                train_df, train_size=min(2048, len(train_df) - 1),
                random_state=SEED, stratify=train_df["label"],
            )
            val_df, _ = train_test_split(
                val_df, train_size=min(512, len(val_df) - 1),
                random_state=SEED, stratify=val_df["label"],
            )
            test_df, _ = train_test_split(
                test_df, train_size=min(1024, len(test_df) - 1),
                random_state=SEED, stratify=test_df["label"],
            )
        self.train_texts = train_df["text"].astype(str).reset_index(drop=True).tolist()
        self.val_texts = val_df["text"].astype(str).reset_index(drop=True).tolist()
        self.train_labels = train_df["label"].to_numpy(dtype=np.int64)
        self.val_labels = val_df["label"].to_numpy(dtype=np.int64)
        self.test_texts = test_df["text"].astype(str).reset_index(drop=True).tolist()
        self.test_labels = test_df["label"].to_numpy(dtype=np.int64)

        # Chỉ token hóa train để đếm tần suất và dựng vocabulary.
        train_tokens = [tokenize(text) for text in self.train_texts]
        counts = Counter(token for tokens in train_tokens for token in tokens)
        self.vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
        available = MAX_VOCAB_SIZE - len(self.vocab)
        for token, count in counts.most_common():
            if count >= MIN_FREQUENCY and len(self.vocab) - 2 < available:
                self.vocab[token] = len(self.vocab)
            if len(self.vocab) >= MAX_VOCAB_SIZE:
                break

        self.train_ids = self._texts_to_ids(self.train_texts)
        self.val_ids = self._texts_to_ids(self.val_texts)
        # Giữ test labels/text nhưng chỉ tokenize khi chuẩn bị final test.
        self.total_train_tokens = sum(map(len, train_tokens))
        self.token_counts = {
            "train": self.total_train_tokens,
            "validation": sum(len(tokenize(text)) for text in self.val_texts),
        }

    def _texts_to_ids(self, texts: list[str]) -> list[list[int]]:
        return [
            [self.vocab.get(token, UNK_ID) for token in tokenize(text)] or [UNK_ID]
            for text in texts
        ]

    def _sequences_and_labels(self, split: str):
        if split == "train":
            return self.train_ids, self.train_labels
        if split == "validation":
            return self.val_ids, self.val_labels
        if split == "test":
            if not hasattr(self, "test_ids"):
                self.test_ids = self._texts_to_ids(self.test_texts)
            return self.test_ids, self.test_labels
        raise ValueError(f"split không hợp lệ: {split}")

    def _loader(self, split: str, max_length: int, shuffle: bool = False) -> DataLoader:
        key = (split, max_length, shuffle)
        # Tạo lại train loader để mỗi cấu hình bắt đầu cùng thứ tự shuffle.
        if not shuffle and key in self._loader_cache:
            return self._loader_cache[key]
        dataset_key = (split, max_length)
        if dataset_key not in self._dataset_cache:
            sequences, labels = self._sequences_and_labels(split)
            self._dataset_cache[dataset_key] = ReviewDataset(sequences, labels, max_length)
        generator = torch.Generator().manual_seed(SEED) if shuffle else None
        loader = DataLoader(
            self._dataset_cache[dataset_key], batch_size=self.batch_size,
            shuffle=shuffle, generator=generator, collate_fn=collate_reviews,
            num_workers=0, pin_memory=self.device.type == "cuda",
        )
        if not shuffle:
            self._loader_cache[key] = loader
        return loader

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        }

    def evaluate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        loss_function = nn.CrossEntropyLoss(reduction="sum")
        with torch.inference_mode():
            for input_ids, labels in loader:
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = model(input_ids)
                total_loss += float(loss_function(logits, labels).item())
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        result = self._classification_metrics(np.asarray(y_true), np.asarray(y_pred))
        result["loss"] = total_loss / len(loader.dataset)
        return result

    def _train(self, model: nn.Module, max_length: int, run_name: str):
        """Train fixed epochs; keep checkpoint with best validation F1."""
        train_loader = self._loader("train", max_length, shuffle=True)
        val_loader = self._loader("validation", max_length)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss(reduction="sum")
        history = []
        best_f1 = -1.0
        best_state = None
        best_epoch = 0

        self._sync()
        start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for input_ids, labels in train_loader:
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss_sum = loss_function(logits, labels)
                (loss_sum / len(labels)).backward()
                optimizer.step()
                train_loss_sum += float(loss_sum.detach().item())

            train_loss = train_loss_sum / len(train_loader.dataset)
            val_metrics = self.evaluate(model, val_loader)
            history.append({
                "experiment": run_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_metrics["loss"],
                "validation_f1": val_metrics["f1"],
            })
            print(
                f"{run_name} | epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        self._sync()
        elapsed = time.perf_counter() - start
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        history_frame = pd.DataFrame(history)
        best_metrics = self.evaluate(model, val_loader)
        self.models[run_name] = model
        self.histories[run_name] = history_frame
        self.metrics[run_name] = best_metrics
        self.times[run_name] = elapsed
        print(f"{run_name}: best validation epoch={best_epoch}, training+validation time={elapsed:.2f}s")
        return model, best_metrics, elapsed, history_frame, best_epoch

    def _new_transformer(self, n_heads: int = 4, max_length: int = MAX_LENGTH,
                         pooling: str = "mean") -> TransformerClassifier:
        return TransformerClassifier(
            vocab_size=len(self.vocab), d_model=D_MODEL, n_heads=n_heads,
            d_ff=D_FF, num_classes=2, pooling=pooling, max_length=max_length,
            n_blocks=N_BLOCKS, dropout=0.1,
        )

    def _add_result(self, **values) -> None:
        base = {
            "experiment": "", "model": "", "eval_split": "validation",
            "run_mode": self.run_mode,
            "n_heads": None, "d_k": None, "d_model": D_MODEL, "d_ff": D_FF,
            "max_length": None, "pooling": None, "parameters": None,
            "transformer_block_parameters": None, "training_time_seconds": None,
            "accuracy": None, "precision": None, "recall": None, "f1": None,
            "tokens_retained_mean": None, "tokens_retained_percent": None,
            "attention_memory_bytes_one_matrix": None, "epoch": None,
            "train_loss": None, "validation_loss": None,
            "attention_entropy_mean": None, "mean_max_attention": None,
            "head_pairwise_similarity_mean": None, "seed": SEED,
        }
        base.update(values)
        self.records.append(base)

    @staticmethod
    def _parameter_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _block_parameter_count(model: TransformerClassifier) -> int:
        return sum(
            p.numel() for block in model.blocks for p in block.parameters()
            if p.requires_grad
        )

    def _metric_row(self, metrics: dict[str, float]) -> dict[str, float]:
        return {key: metrics[key] for key in ("accuracy", "precision", "recall", "f1")}

    def run_heads(self) -> pd.DataFrame:
        """Bài 19: train one independent model per head configuration."""
        rows = []
        for n_heads in (1, 2, 4, 8):
            set_seed(SEED)
            name = f"heads_{n_heads}"
            model, metrics, seconds, _, _ = self._train(
                self._new_transformer(n_heads=n_heads, max_length=MAX_LENGTH),
                MAX_LENGTH, name,
            )
            row = {
                "Heads": n_heads, "d_k": D_MODEL // n_heads,
                "Classifier parameters": self._parameter_count(model),
                "TransformerBlock parameters": self._block_parameter_count(model),
                "Training time (s)": seconds, "Validation accuracy": metrics["accuracy"],
                "Validation F1": metrics["f1"], "Best validation loss": metrics["loss"],
            }
            rows.append(row)
            self._add_result(
                experiment="Bài 19 — Attention heads", model=name,
                n_heads=n_heads, d_k=D_MODEL // n_heads, max_length=MAX_LENGTH,
                pooling="mean", parameters=row["Classifier parameters"],
                transformer_block_parameters=row["TransformerBlock parameters"],
                training_time_seconds=seconds, **self._metric_row(metrics),
                validation_loss=metrics["loss"],
            )
        return pd.DataFrame(rows)

    def run_head_diversity(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Bài 20: lấy attention weights thật từ model 4-head đã train."""
        model = self.models["heads_4"]
        sample_index = min(range(len(self.val_ids)), key=lambda i: len(self.val_ids[i]))
        # Cắt prefix ngắn để heatmap dễ đọc; đây vẫn là đầu vào thật từ validation.
        ids = self.val_ids[sample_index][:min(MAX_LENGTH, 32)]
        tokens = tokenize(self.val_texts[sample_index])[:len(ids)]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        model.eval()
        with torch.inference_mode():
            _, weights = model(input_ids, return_attention=True)
        matrices = weights[0, :, :len(ids), :len(ids)].cpu().numpy()

        entropies, mean_maxima = [], []
        for matrix in matrices:
            entropy = -(matrix * np.log(np.clip(matrix, 1e-12, 1.0))).sum(axis=-1).mean()
            entropies.append(float(entropy))
            mean_maxima.append(float(matrix.max(axis=-1).mean()))
        flattened = matrices.reshape(matrices.shape[0], -1)
        norms = np.linalg.norm(flattened, axis=1, keepdims=True).clip(min=1e-12)
        similarity = (flattened / norms) @ (flattened / norms).T
        stats = pd.DataFrame({
            "Head": np.arange(1, len(matrices) + 1),
            "Mean attention entropy": entropies,
            "Mean maximum weight": mean_maxima,
        })
        pairwise = pd.DataFrame(
            similarity, index=[f"Head {i}" for i in range(1, 5)],
            columns=[f"Head {i}" for i in range(1, 5)],
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        image = None
        for head, axis in enumerate(axes.flat):
            image = axis.imshow(matrices[head], cmap="viridis", vmin=0.0, vmax=1.0)
            axis.set_title(f"Head {head + 1}")
            axis.set_xlabel("Key token")
            axis.set_ylabel("Query token")
            axis.set_xticks(range(len(tokens)), labels=tokens, rotation=90)
            axis.set_yticks(range(len(tokens)), labels=tokens)
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Attention weight", shrink=0.8)
        path = self.output_dir / "attention_heads.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)

        for index, row in stats.iterrows():
            self._add_result(
                experiment="Bài 20 — Attention diversity",
                model=f"heads_4_head_{index + 1}", n_heads=4, d_k=32,
                max_length=MAX_LENGTH, pooling="mean",
                attention_entropy_mean=row["Mean attention entropy"],
                mean_max_attention=row["Mean maximum weight"],
                head_pairwise_similarity_mean=float(
                    np.delete(similarity[index], index).mean()
                ),
            )
        print("Attention sentence:", " ".join(tokens))
        print("Saved:", path)
        return stats, pairwise

    def run_lstm(self) -> pd.DataFrame:
        """Train LAB 05-style packed LSTM on the same split and vocabulary."""
        set_seed(SEED)
        model = LSTMClassifier(len(self.vocab), hidden_size=D_MODEL, dropout=0.1)
        model, metrics, seconds, _, _ = self._train(model, MAX_LENGTH, "lstm")
        row = {
            "Model": "LSTM", "Parameters": self._parameter_count(model),
            "Training time (s)": seconds, "Validation accuracy": metrics["accuracy"],
            "Validation F1": metrics["f1"], "Best validation loss": metrics["loss"],
        }
        self._add_result(
            experiment="Bài 21 — LSTM vs Transformer", model="LSTM",
            max_length=MAX_LENGTH, pooling="final hidden state",
            parameters=row["Parameters"], training_time_seconds=seconds,
            **self._metric_row(metrics), validation_loss=metrics["loss"],
        )
        transformer = self.models["heads_4"]
        tmetrics = self.metrics["heads_4"]
        trow = {
            "Model": "Transformer (4 heads, mean)",
            "Parameters": self._parameter_count(transformer),
            "Training time (s)": self.times["heads_4"],
            "Validation accuracy": tmetrics["accuracy"], "Validation F1": tmetrics["f1"],
            "Best validation loss": tmetrics["loss"],
        }
        self._add_result(
            experiment="Bài 21 — LSTM vs Transformer", model="Transformer (4 heads, mean)",
            n_heads=4, d_k=32, max_length=MAX_LENGTH, pooling="mean",
            parameters=trow["Parameters"],
            transformer_block_parameters=self._block_parameter_count(transformer),
            training_time_seconds=trow["Training time (s)"],
            **self._metric_row(tmetrics), validation_loss=tmetrics["loss"],
        )
        return pd.DataFrame([trow, row])

    def run_sequence_lengths(self) -> pd.DataFrame:
        """Bài 22: giữ mọi biến cố định, tái dùng len=128 run từ Bài 19."""
        rows = []
        for max_length in MAX_LENGTHS:
            if max_length == MAX_LENGTH:
                model = self.models["heads_4"]
                metrics = self.metrics["heads_4"]
                seconds = self.times["heads_4"]
            else:
                set_seed(SEED)
                name = f"length_{max_length}"
                model, metrics, seconds, _, _ = self._train(
                    self._new_transformer(n_heads=4, max_length=max_length),
                    max_length, name,
                )
            retained = sum(min(len(ids), max_length) for ids in self.train_ids)
            mean_retained = retained / len(self.train_ids)
            retained_percent = 100.0 * retained / max(1, self.total_train_tokens)
            attention_bytes = self.batch_size * 4 * max_length * max_length * 4
            row = {
                "Length": max_length, "Mean tokens retained/sample": mean_retained,
                "Train tokens retained (%)": retained_percent,
                "One attention matrix (bytes)": attention_bytes,
                "Training time (s)": seconds, "Validation F1": metrics["f1"],
                "Validation accuracy": metrics["accuracy"],
            }
            rows.append(row)
            self._add_result(
                experiment="Bài 22 — Sequence length", model=f"Transformer length {max_length}",
                n_heads=4, d_k=32, max_length=max_length, pooling="mean",
                parameters=self._parameter_count(model),
                transformer_block_parameters=self._block_parameter_count(model),
                training_time_seconds=seconds, **self._metric_row(metrics),
                tokens_retained_mean=mean_retained,
                tokens_retained_percent=retained_percent,
                attention_memory_bytes_one_matrix=attention_bytes,
                validation_loss=metrics["loss"],
            )
        frame = pd.DataFrame(rows)
        theory = {int(length): int(length ** 2) for length in frame["Length"]}
        print("Attention elements T^2:", theory)
        return frame

    def run_pooling(self) -> pd.DataFrame:
        """Bài 25: so mean pooling với learned [CLS] vector pooling."""
        mean_model = self.models["heads_4"]
        mean_metrics = self.metrics["heads_4"]
        mean_time = self.times["heads_4"]
        set_seed(SEED)
        cls_model = self._new_transformer(n_heads=4, max_length=MAX_LENGTH, pooling="cls")
        cls_model, cls_metrics, cls_time, _, _ = self._train(cls_model, MAX_LENGTH, "pooling_cls")
        rows = []
        for pooling, model, metrics, seconds in (
            ("Mean", mean_model, mean_metrics, mean_time),
            ("CLS-style (learned token)", cls_model, cls_metrics, cls_time),
        ):
            row = {
                "Pooling": pooling, "Parameters": self._parameter_count(model),
                "Validation accuracy": metrics["accuracy"],
                "Validation F1": metrics["f1"], "Training time (s)": seconds,
            }
            rows.append(row)
            self._add_result(
                experiment="Bài 25 — Pooling strategy", model=f"Transformer {pooling}",
                n_heads=4, d_k=32, max_length=MAX_LENGTH, pooling=pooling,
                parameters=row["Parameters"],
                transformer_block_parameters=self._block_parameter_count(model),
                training_time_seconds=seconds, **self._metric_row(metrics),
                validation_loss=metrics["loss"],
            )
        return pd.DataFrame(rows)

    def save_learning_curves(self) -> pd.DataFrame:
        """Bài 29: xuất loss theo epoch cho Transformer và LSTM thực tế."""
        history = pd.concat(
            [self.histories["heads_4"], self.histories["lstm"]], ignore_index=True
        )
        history.to_csv(self.output_dir / "learning_curves.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for axis, run_name, title in (
            (axes[0], "heads_4", "Transformer (4 heads, mean)"),
            (axes[1], "lstm", "LSTM"),
        ):
            part = self.histories[run_name]
            axis.plot(part["epoch"], part["train_loss"], marker="o", label="Train loss")
            axis.plot(part["epoch"], part["validation_loss"], marker="s", label="Validation loss")
            axis.set(title=title, xlabel="Epoch", ylabel="Loss")
            axis.set_xticks(range(1, self.epochs + 1))
            axis.grid(alpha=0.25)
            axis.legend()
        path = self.output_dir / "learning_curves.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        for _, row in history.iterrows():
            self._add_result(
                experiment="Bài 29 — Learning curves", model=row["experiment"],
                epoch=int(row["epoch"]), train_loss=float(row["train_loss"]),
                validation_loss=float(row["validation_loss"]),
            )
        print("Saved:", path, self.output_dir / "learning_curves.csv")
        return history

    def final_test_evaluation(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Đánh giá LSTM và baseline Transformer trên toàn test gốc một lần."""
        test_loader = self._loader("test", MAX_LENGTH)
        rows = []
        predictions_by_model = {}
        for run_name, label in (("heads_4", "Transformer (4 heads, mean)"), ("lstm", "LSTM")):
            model = self.models[run_name]
            model.eval()
            y_true, y_pred = [], []
            with torch.inference_mode():
                for input_ids, labels in test_loader:
                    logits = model(input_ids.to(self.device, non_blocking=True))
                    y_true.extend(labels.numpy().tolist())
                    y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_true = np.asarray(y_true, dtype=np.int64)
            y_pred = np.asarray(y_pred, dtype=np.int64)
            metrics = self._classification_metrics(y_true, y_pred)
            predictions_by_model[run_name] = y_pred
            model_row = {
                "Model": label, "Parameters": self._parameter_count(model),
                "Training time (s)": self.times[run_name],
                "Accuracy": metrics["accuracy"], "Precision": metrics["precision"],
                "Recall": metrics["recall"], "F1": metrics["f1"],
            }
            rows.append(model_row)
            self._add_result(
                experiment="Final test evaluation", model=label,
                eval_split="test" if self.run_mode == "full" else "smoke_test_subset",
                n_heads=4 if run_name == "heads_4" else None,
                d_k=32 if run_name == "heads_4" else None,
                max_length=MAX_LENGTH, pooling="mean" if run_name == "heads_4" else "final hidden state",
                parameters=self._parameter_count(model),
                transformer_block_parameters=(
                    self._block_parameter_count(model) if run_name == "heads_4" else None
                ),
                training_time_seconds=self.times[run_name], **metrics,
            )

        transformer_pred = predictions_by_model["heads_4"]
        matrix = confusion_matrix(self.test_labels, transformer_pred, labels=[0, 1])
        if matrix.sum() != len(self.test_labels):
            raise AssertionError("Confusion matrix sum không bằng số test samples")
        fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Negative (0)", "Positive (1)"])
        ax.set_yticks([0, 1], labels=["Negative (0)", "Positive (1)"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("IMDb Test — Transformer (4 heads, mean pooling)")
        threshold = matrix.max() / 2
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:,}", ha="center", va="center",
                    color="white" if value > threshold else "black")
        fig.colorbar(image, ax=ax, label="Number of reviews")
        path = self.output_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return pd.DataFrame(rows), self.test_labels.copy(), transformer_pred

    def predict_examples(self) -> pd.DataFrame:
        """Chạy năm loại câu yêu cầu trên classifier Transformer đã train."""
        examples = {
            "Short": "I loved this movie.",
            "Long": ("The opening was promising and the actors were convincing, but the plot became "
                     "repetitive, the middle dragged, and the ending was disappointing despite a few "
                     "beautiful scenes and one excellent performance."),
            "Negation": "I did not enjoy this film.",
            "Contrast": "This movie has a good beginning but the ending is terrible.",
            "Ambiguous": "The movie was not bad, although it was not especially memorable.",
        }
        model = self.models["heads_4"]
        rows = []
        for kind, text in examples.items():
            ids = [self.vocab.get(token, UNK_ID) for token in tokenize(text)] or [UNK_ID]
            retained = ids[:MAX_LENGTH]
            input_ids = torch.tensor([retained], dtype=torch.long, device=self.device)
            model.eval()
            with torch.inference_mode():
                probabilities = torch.softmax(model(input_ids), dim=1)[0].cpu().numpy()
            pred = int(probabilities.argmax())
            rows.append({
                "Type": kind, "Input text": text,
                "Predicted label": "Positive" if pred == 1 else "Negative",
                "P(Negative)": float(probabilities[0]), "P(Positive)": float(probabilities[1]),
                "Original tokens": len(ids), "Retained tokens": len(retained),
            })
        return pd.DataFrame(rows)

    def export_results(self) -> pd.DataFrame:
        """Lưu các metrics/history đã đo, giữ ô không áp dụng rỗng."""
        results = pd.DataFrame(self.records)
        path = self.output_dir / "results.csv"
        results.to_csv(path, index=False)
        print(f"Saved {len(results)} measured rows to {path}")
        return results

    def environment_summary(self) -> dict[str, object]:
        return {
            "torch": torch.__version__,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "dataset": "IMDb labeled train/test; unsupervised excluded",
            "split": f"train={len(self.train_ids)}, validation={len(self.val_ids)}, test={len(self.test_labels)}",
            "split_seed": SPLIT_SEED,
            "model_seed": SEED,
            "vocabulary_size_including_PAD_UNK": len(self.vocab),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "run_mode": self.run_mode,
            "learning_rate": LEARNING_RATE,
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "transformer_blocks": N_BLOCKS,
            "max_length_default": MAX_LENGTH,
            "positive_label": 1,
        }
