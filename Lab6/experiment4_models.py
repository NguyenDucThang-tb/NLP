"""LAB 06 Experiment 4: compare LSTM and masked self-attention context."""

from __future__ import annotations

import csv
import html
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


PAD_ID = 0
UNK_ID = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEED = 42
SPLIT_SEED = 2026
EMBEDDING_DIM = 50
HIDDEN_SIZE = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3
MAX_LENGTH = 100
MAX_VOCAB_SIZE = 20_000
MIN_FREQ = 2


def set_seed(seed: int = SEED) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms aid repeatability where the backend supports them.
    torch.use_deterministic_algorithms(True, warn_only=True)


def tokenize(text: str) -> list[str]:
    """Match the LAB 05 lowercase, HTML cleanup and word-tokenization steps."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def build_vocabulary(train_texts: list[str]) -> dict[str, int]:
    """Build a capped vocabulary from training text only, including PAD/UNK."""
    counts = Counter()
    for text in train_texts:
        counts.update(tokenize(text))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, count in counts.most_common():
        if count >= MIN_FREQ and len(vocab) < MAX_VOCAB_SIZE:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> tuple[list[int], int]:
    """Convert text to IDs and apply the LAB 05 right truncation/padding."""
    ids = [vocab.get(token, UNK_ID) for token in tokenize(text)][:MAX_LENGTH]
    if not ids:
        ids = [UNK_ID]
    length = len(ids)
    return ids + [PAD_ID] * (MAX_LENGTH - length), length


def make_loader(frame: pd.DataFrame, vocab: dict[str, int], shuffle: bool = False) -> DataLoader:
    encoded = [encode_text(text, vocab) for text in frame["text"].astype(str)]
    ids = torch.tensor([row[0] for row in encoded], dtype=torch.long)
    lengths = torch.tensor([row[1] for row in encoded], dtype=torch.long)
    labels = torch.tensor(frame["label"].to_numpy(dtype=np.float32), dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(
        TensorDataset(ids, lengths, labels), batch_size=BATCH_SIZE,
        shuffle=shuffle, generator=generator,
    )


class LSTMClassifier(nn.Module):
    """Embedding -> packed one-layer LSTM -> final hidden -> binary logit."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        representation = hidden[-1]
        logits = self.classifier(representation).squeeze(-1)
        return logits, representation


class AttentionClassifier(nn.Module):
    """Embedding -> single-head scaled self-attention -> masked mean -> logit."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_ID)
        self.query = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.key = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.classifier = nn.Linear(EMBEDDING_DIM, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embedding(input_ids)
        batch_size, seq_len, _ = embedded.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        valid_mask = positions < lengths.unsqueeze(1)
        if torch.any(lengths < 1):
            raise ValueError("Mỗi sequence cần ít nhất một token hợp lệ")

        query = self.query(embedded)
        key = self.key(embedded)
        value = self.value(embedded)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(EMBEDDING_DIM)

        # Chỉ PAD ở các key positions bị loại khỏi softmax.
        key_mask = valid_mask.unsqueeze(1)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)

        # Padded queries không tham gia context hay masked pooling.
        weights = weights * valid_mask.unsqueeze(-1).to(weights.dtype)
        contextual = torch.matmul(weights, value)
        contextual = contextual * valid_mask.unsqueeze(-1).to(contextual.dtype)
        representation = contextual.sum(dim=1) / lengths.unsqueeze(1).to(contextual.dtype)
        logits = self.classifier(representation).squeeze(-1)
        return logits, representation, weights


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_dataset_and_splits():
    """Reuse IMDb and exactly the LAB 05 split recipe (80/20 of train, official test)."""
    dataset = load_dataset("imdb")
    source_train = dataset["train"].to_pandas()[["text", "label"]].copy()
    source_test = dataset["test"].to_pandas()[["text", "label"]].copy()
    source_train["source_id"] = np.arange(len(source_train), dtype=np.int64)
    source_test["source_id"] = np.arange(len(source_test), dtype=np.int64)
    source_train["source_split"] = "train"
    source_test["source_split"] = "test"
    train_data, val_data = train_test_split(
        source_train, test_size=0.20, random_state=SPLIT_SEED,
        stratify=source_train["label"],
    )
    return (
        train_data.reset_index(drop=True), val_data.reset_index(drop=True),
        source_test.reset_index(drop=True), len(source_train),
    )


def _evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    truths, probabilities = [], []
    attention_example = None
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids, lengths = input_ids.to(device), lengths.to(device)
            labels = labels.to(device)
            output = model(input_ids, lengths)
            logits = output[0]
            total_loss += criterion(logits, labels).item() * len(labels)
            truths.extend(labels.cpu().numpy().astype(np.int64).tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    y_true = np.asarray(truths, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    predictions = (probs >= 0.5).astype(np.int64)
    metrics = {
        "Loss": total_loss / len(loader.dataset),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "F1": float(f1_score(y_true, predictions, average="binary", pos_label=1, zero_division=0)),
        "Macro-F1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }
    return metrics, y_true, predictions, probs


def _train(model, train_loader, val_loader, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)[0]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_training_seconds = time.perf_counter() - epoch_start
        train_loss = total_loss / len(train_loader.dataset)
        val_metrics, _, _, _ = _evaluate(model, val_loader, device)
        history.append({
            "Epoch": epoch, "Train Loss": train_loss,
            "Validation Loss": val_metrics["Loss"],
            "Validation Accuracy": val_metrics["Accuracy"],
            "Validation F1": val_metrics["F1"],
            "Validation Macro-F1": val_metrics["Macro-F1"],
            "Epoch Training Time (s)": epoch_training_seconds,
        })
        print(
            f"{model.__class__.__name__} epoch {epoch}/{EPOCHS}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['Loss']:.4f}, "
            f"val_f1={val_metrics['F1']:.4f}, train_seconds={epoch_training_seconds:.2f}"
        )
        if val_metrics["F1"] > best_f1:
            best_f1, best_epoch = val_metrics["F1"], epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - total_start
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), training_seconds, best_epoch


def _save_confusion_matrix(path: Path, y_true, y_pred, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Negative (0)", "Positive (1)"])
    ax.set_yticks([0, 1], ["Negative (0)", "Positive (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(histories: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for column, filename, title, ylabel in [
        ("Train Loss", "experiment4_training_loss.png", "Experiment 4 Training Loss", "Loss"),
        ("Validation F1", "experiment4_validation_f1.png", "Experiment 4 Validation F1", "F1 score"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model_name, history in histories.items():
            ax.plot(history["Epoch"], history[column], marker="o", label=model_name)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, xticks=range(1, EPOCHS + 1))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def _save_attention_inspection(model, test_data, vocab, device, output_dir):
    lengths_all = [len(tokenize(text)) for text in test_data["text"].astype(str)]
    candidates = [i for i, length in enumerate(lengths_all) if 8 <= length <= 30]
    sample_index = candidates[0] if candidates else int(np.argmin(np.abs(np.asarray(lengths_all) - 16)))
    text = str(test_data.iloc[sample_index]["text"])
    tokens = tokenize(text)[:MAX_LENGTH]
    ids, length = encode_text(text, vocab)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    length_tensor = torch.tensor([length], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits, representation, weights = model(input_tensor, length_tensor)
        probability = float(torch.sigmoid(logits)[0].cpu())
        prediction = int(probability >= 0.5)
        matrix = weights[0, :length, :length].cpu().numpy()
        pooled_shape = tuple(representation.shape)

    matrix_path = output_dir / "experiment4_attention_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Query token / Key token", *tokens])
        for token, row in zip(tokens, matrix):
            writer.writerow([token, *(f"{value:.12f}" for value in row)])

    fig, ax = plt.subplots(figsize=(max(7, len(tokens) * 0.65), max(6, len(tokens) * 0.55)))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(1.0, float(matrix.max())))
    ax.set_xticks(np.arange(length), tokens, rotation=45, ha="right")
    ax.set_yticks(np.arange(length), tokens)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    ax.set_title("Trained Attention Classifier — Test Sample")
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    heatmap_path = output_dir / "experiment4_trained_attention_heatmap.png"
    fig.savefig(heatmap_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    sample = {
        "Test row index": sample_index,
        "Ground truth": int(test_data.iloc[sample_index]["label"]),
        "Attention prediction": prediction,
        "Attention P(positive)": probability,
        "Attention context shape": str(pooled_shape),
        "Token count shown": length,
        "Text": text,
        "Tokens": " ".join(tokens),
    }
    return sample, matrix, matrix_path, heatmap_path


def run_experiment4(output_dir: str | Path, lab5_dir: str | Path) -> dict:
    """Train/evaluate two classifiers on the LAB 05 IMDb split and save artifacts."""
    output_dir, lab5_dir = Path(output_dir), Path(lab5_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab5_config_path = lab5_dir / "results" / "rnn_lstm_gru_config.json"
    lab5_lstm_path = lab5_dir / "models" / "rnn_lstm_gru" / "lstm_best.pt"
    lab5_history_path = lab5_dir / "results" / "rnn_lstm_gru_lstm_history.csv"
    for required_file in (lab5_dir / "sequence_classification.ipynb", lab5_config_path, lab5_lstm_path, lab5_history_path):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required LAB 05 reference file missing: {required_file}")
    lab5_config = json.loads(lab5_config_path.read_text(encoding="utf-8"))
    expected_lab5_config = {
        "embedding_dim": EMBEDDING_DIM, "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS, "max_length": MAX_LENGTH,
        "vocabulary_size": MAX_VOCAB_SIZE,
    }
    if any(lab5_config.get(key) != value for key, value in expected_lab5_config.items()):
        raise ValueError(f"Experiment configuration differs from LAB 05: {lab5_config}")
    lab5_history = pd.read_csv(lab5_history_path)
    lab5_checkpoint = torch.load(lab5_lstm_path, map_location="cpu", weights_only=False)
    if lab5_checkpoint.get("model_type") != "LSTM" or lab5_checkpoint.get("vocab_size") != MAX_VOCAB_SIZE:
        raise ValueError("Existing LAB 05 LSTM checkpoint does not match the recorded config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    # Preserve every existing Experiment 1-3 artifact and compare after E4.
    previous_names = [
        "experiment1_attention_distribution.csv", "experiment1_attention_distribution.png",
        "experiment2_query_comparison.csv", "experiment2_query_comparison.png",
        "experiment2_query_heatmap.png", "experiment3_attention_heatmap_sentence1.png",
        "experiment3_attention_heatmap_sentence2.png", "experiment3_selected_tokens_comparison.png",
        "experiment3_attention_matrix_sentence1.csv", "experiment3_attention_matrix_sentence2.csv",
        "experiment3_selected_token_analysis.csv",
    ]
    previous_bytes = {name: (output_dir / name).read_bytes() for name in previous_names}

    train_data, val_data, test_data, original_train_count = get_dataset_and_splits()
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    if all_data[["text", "label"]].isna().any().any():
        raise ValueError("IMDb có text/label thiếu; không thể áp dụng pipeline LAB 05 nguyên trạng")
    if set(all_data["label"].unique()) != {0, 1}:
        raise ValueError(f"Label space không như mong đợi: {sorted(all_data['label'].unique())}")
    if set(train_data["source_id"]) & set(val_data["source_id"]):
        raise AssertionError("Train/validation overlap")
    if len(train_data) + len(val_data) != original_train_count:
        raise AssertionError("Train/validation split không bao phủ IMDb train split")

    vocab = build_vocabulary(train_data["text"].astype(str).tolist())
    train_loader = make_loader(train_data, vocab, shuffle=True)
    val_loader = make_loader(val_data, vocab)
    test_loader = make_loader(test_data, vocab)
    y_test = test_data["label"].to_numpy(dtype=np.int64)
    histories, models, rows, predictions = {}, {}, [], {}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    model_specs = [("LSTM", LSTMClassifier), ("Attention", AttentionClassifier)]
    for model_name, model_class in model_specs:
        set_seed(SEED)
        model = model_class(len(vocab)).to(device)
        parameter_count = count_parameters(model)
        # Each model receives a fresh loader with the same initial batch order.
        model_train_loader = make_loader(train_data, vocab, shuffle=True)
        model, history, training_seconds, best_epoch = _train(
            model, model_train_loader, val_loader, device
        )
        test_metrics, y_observed, y_pred, y_prob = _evaluate(model, test_loader, device)
        if not np.array_equal(y_observed, y_test):
            raise AssertionError(f"Test ordering/labels differ for {model_name}")

        models[model_name] = model
        histories[model_name] = history
        predictions[model_name] = (y_pred, y_prob)
        history.insert(0, "Model", model_name)
        rows.append({
            "Model": model_name,
            "Accuracy": test_metrics["Accuracy"],
            "Precision": float(precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)),
            "F1": test_metrics["F1"],
            "Macro-F1": test_metrics["Macro-F1"],
            "Training Time (s)": training_seconds,
            "Parameters": parameter_count,
            "Epochs Trained": EPOCHS,
            "Best Epoch (Validation F1)": best_epoch,
        })
        torch.save({
            "model": model.state_dict(), "model_name": model_name,
            "vocabulary": vocab, "config": {
                "seed": SEED, "split_seed": SPLIT_SEED,
                "embedding_dim": EMBEDDING_DIM, "hidden_size": HIDDEN_SIZE,
                "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS, "max_length": MAX_LENGTH,
                "vocabulary_size": len(vocab), "device": str(device),
            },
        }, checkpoint_dir / f"experiment4_{model_name.lower()}.pt")
        print(f"{model_name}: best validation-F1 epoch={best_epoch}; test metrics={test_metrics}")

    results = pd.DataFrame(rows)
    history_all = pd.concat(histories.values(), ignore_index=True)
    results_path = output_dir / "experiment4_results.csv"
    history_path = output_dir / "experiment4_training_history.csv"
    results.to_csv(results_path, index=False)
    history_all.to_csv(history_path, index=False)
    _plot_history(histories, output_dir)

    confusion_paths = {}
    for model_name, (y_pred, _) in predictions.items():
        path = output_dir / f"experiment4_confusion_matrix_{model_name.lower()}.png"
        _save_confusion_matrix(path, y_test, y_pred, f"{model_name} — IMDb test split")
        confusion_paths[model_name] = path

    # Compare output shape and predictions on the same, fixed test review.
    inspection, attention_matrix, attention_matrix_path, attention_heatmap_path = _save_attention_inspection(
        models["Attention"], test_data, vocab, device, output_dir
    )
    sample_index = inspection["Test row index"]
    sample_ids, sample_length = encode_text(str(test_data.iloc[sample_index]["text"]), vocab)
    sample_input = torch.tensor([sample_ids], dtype=torch.long, device=device)
    sample_length_tensor = torch.tensor([sample_length], dtype=torch.long, device=device)
    models["LSTM"].eval()
    with torch.no_grad():
        lstm_logits, lstm_representation = models["LSTM"](sample_input, sample_length_tensor)
        lstm_probability = float(torch.sigmoid(lstm_logits)[0].cpu())
    inspection.update({
        "LSTM prediction": int(lstm_probability >= 0.5),
        "LSTM P(positive)": lstm_probability,
        "LSTM final hidden shape": str(tuple(lstm_representation.shape)),
    })
    representation_path = output_dir / "experiment4_representation_inspection.csv"
    pd.DataFrame([inspection]).to_csv(representation_path, index=False)

    config = {
        "dataset": "IMDb (Hugging Face imdb; train and test labeled splits only)",
        "original_train_samples": original_train_count,
        "train_samples": len(train_data), "validation_samples": len(val_data),
        "test_samples": len(test_data), "num_classes": 2,
        "label_mapping": {"0": "Negative", "1": "Positive"},
        "split": "IMDb train split stratified 80/20 with seed 2026; official IMDb test retained",
        "lab5_reference_files": [
            str(lab5_dir / "sequence_classification.ipynb"), str(lab5_config_path),
            str(lab5_lstm_path), str(lab5_history_path),
        ],
        "lab5_original_lstm_checkpoint_reused": False,
        "lab5_checkpoint_note": "Architecture/config inspected; E4 LSTM retrained with seed 42 for matched E4 initialization/training conditions.",
        "lab5_lstm_history_epochs": int(len(lab5_history)),
        "seed": SEED, "split_seed": SPLIT_SEED, "device": str(device),
        "python_version": __import__("sys").version.split()[0],
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "vocabulary_size_including_PAD_UNK": len(vocab),
        "tokenizer": "lowercase, HTML unescape/tag cleanup, regex [a-z0-9]+(?:'[a-z]+)?",
        "max_length": MAX_LENGTH, "truncation": "keep first 100 tokens; right pad with PAD_ID",
        "embedding_dim": EMBEDDING_DIM, "lstm_hidden_size": HIDDEN_SIZE,
        "attention_qk_v_dim": EMBEDDING_DIM, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "optimizer": "Adam", "epochs": EPOCHS,
        "loss": "BCEWithLogitsLoss", "early_stopping": False,
        "checkpoint_selection": "best validation binary F1, positive label 1",
        "attention": "single-head scaled dot-product; no positional encoding; masked mean pooling",
        "lab5_source_not_modified": True,
    }
    config_path = output_dir / "experiment4_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Executable checks on data, padding/masking, gradients, metrics and artifacts.
    tests = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(f"FAIL: {name}")
        tests.append(name)
        print(f"PASS: {name}")

    identity_keys = [(row.source_split, int(row.source_id)) for row in all_data.itertuples()]
    check("same dataset splits and no sample identity overlap",
          len(train_data) + len(val_data) + len(test_data) == 50_000 and
          len(set(identity_keys)) == len(all_data) and
          not (set(train_data.source_id) & set(val_data.source_id)))
    check("same tokenizer, train-only vocabulary, max length and labels",
          len(vocab) <= MAX_VOCAB_SIZE and vocab.get(PAD_TOKEN) == PAD_ID and
          vocab.get(UNK_TOKEN) == UNK_ID and set(y_test.tolist()) == {0, 1})

    ids, length = encode_text("A short review", vocab)
    toy_ids = torch.tensor([ids], dtype=torch.long, device=device)
    toy_lengths = torch.tensor([length], dtype=torch.long, device=device)
    for model in models.values():
        model.eval()
        with torch.no_grad():
            output = model(toy_ids, toy_lengths)
        check("model input/output shapes",
              output[0].shape == (1,) and output[1].shape[0] == 1 and
              torch.isfinite(output[0]).all() and torch.isfinite(output[1]).all())

    # Invariance to extra trailing PAD verifies LSTM uses packed true length.
    toy_embedding = models["LSTM"].embedding(toy_ids)
    packed_a = pack_padded_sequence(toy_embedding, toy_lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, state_a = models["LSTM"].lstm(packed_a)
    longer_ids = torch.cat([toy_ids, torch.zeros((1, 7), dtype=torch.long, device=device)], dim=1)
    longer_embedding = models["LSTM"].embedding(longer_ids)
    packed_b = pack_padded_sequence(longer_embedding, toy_lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, state_b = models["LSTM"].lstm(packed_b)
    check("LSTM final state excludes trailing PAD", torch.allclose(state_a[0], state_b[0], atol=1e-6))

    # Attention mask and normalization, including a deliberately padded toy input.
    attention_test_ids = torch.tensor([[vocab.get("a", UNK_ID), vocab.get("movie", UNK_ID), PAD_ID, PAD_ID]], device=device)
    attention_test_lengths = torch.tensor([2], device=device)
    with torch.no_grad():
        _, pooled_test, weights_test = models["Attention"](attention_test_ids, attention_test_lengths)
    check("padded keys receive zero weight and padded queries are excluded",
          torch.all(weights_test[:, :, 2:] == 0) and torch.all(weights_test[:, 2:, :] == 0) and
          torch.isfinite(pooled_test).all())
    check("attention weights normalize over valid keys",
          torch.allclose(weights_test[:, :2, :2].sum(-1), torch.ones((1, 2), device=device), atol=1e-6))

    for name, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(toy_ids, toy_lengths)[0]
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0], device=device))
        loss.backward()
        essential = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        check(f"{name} gradient flow is finite", all(grad is not None and torch.isfinite(grad).all() for grad in essential))
        model.eval()

    check("finite logits/loss/attention values",
          all(np.isfinite(value) for value in results[["Accuracy", "F1", "Training Time (s)" ]].to_numpy().ravel()) and
          np.isfinite(attention_matrix).all())
    check("predictions are in label space", all(set(pred.tolist()) <= {0, 1} for pred, _ in predictions.values()))
    for model_name, (pred, _) in predictions.items():
        row = results.loc[results.Model == model_name].iloc[0]
        check(f"{model_name} metrics use the actual test labels",
              np.isclose(row.Accuracy, accuracy_score(y_test, pred)) and
              np.isclose(row.F1, f1_score(y_test, pred, average="binary", pos_label=1, zero_division=0)))
        check(f"{model_name} parameter count is from the model",
              int(row.Parameters) == count_parameters(models[model_name]))
        check(f"{model_name} training time is measured and positive", float(row["Training Time (s)"]) > 0)

    artifact_paths = [
        results_path, history_path, output_dir / "experiment4_training_loss.png",
        output_dir / "experiment4_validation_f1.png", *confusion_paths.values(),
        attention_matrix_path, attention_heatmap_path, representation_path, config_path,
        checkpoint_dir / "experiment4_lstm.pt", checkpoint_dir / "experiment4_attention.pt",
    ]
    check("all result files exist and are non-empty", all(p.is_file() and p.stat().st_size > 0 for p in artifact_paths))
    check("fixed seed produces identical model initialization",
          _same_initialization(len(vocab)))
    check("Experiment 1-3 artifacts remain byte-identical",
          all((output_dir / name).read_bytes() == data for name, data in previous_bytes.items()))

    test_report_path = output_dir / "experiment4_test_report.txt"
    test_report_path.write_text("\n".join(f"PASS: {name}" for name in tests) + "\n", encoding="utf-8")
    return {
        "results": results, "histories": history_all, "models": models,
        "predictions": predictions, "vocabulary": vocab, "config": config,
        "files": artifact_paths + [test_report_path], "test_names": tests,
        "test_data": test_data, "y_test": y_test,
    }


def _same_initialization(vocab_size: int) -> bool:
    initial_states = []
    for model_class in (LSTMClassifier, AttentionClassifier):
        set_seed(SEED)
        first = model_class(vocab_size).state_dict()
        set_seed(SEED)
        second = model_class(vocab_size).state_dict()
        initial_states.append(all(torch.equal(first[key], second[key]) for key in first))
    return all(initial_states)
