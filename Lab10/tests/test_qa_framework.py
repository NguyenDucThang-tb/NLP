"""Tests for the QA framework and implemented student algorithms."""

from __future__ import annotations

import unittest
import csv
import tempfile
from pathlib import Path

from qa_model import (
    QAConfig,
    _balanced_title_partition,
    create_fixed_splits,
    keyword_overlap_baseline,
    environment_report,
    upsert_results_csv,
    validate_split_titles,
    split_context_sentences,
)


class TestConfiguration(unittest.TestCase):
    def test_common_stride_is_valid_for_context_length_experiment(self):
        for max_length in (128, 256, 384):
            config = QAConfig(max_length=max_length, stride=64)
            config.validate()
            self.assertEqual(config.stride, 64)

    def test_default_configuration(self):
        config = QAConfig()
        config.validate()
        self.assertEqual(config.mode, "QUICK")
        self.assertEqual(config.sample_limits(), (64, 16, 16))


class TestDataSplit(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": str(i), "title": f"Article {i // 3}", "context": f"context {i}",
             "question": f"question {i}", "answers": {"text": ["answer"], "answer_start": [0]}}
            for i in range(30)
        ]

    def test_title_groups_do_not_cross_validation_and_test(self):
        validation, test = _balanced_title_partition(self.rows, seed=42)
        val_titles = {row["title"] for row in validation}
        test_titles = {row["title"] for row in test}
        self.assertFalse(val_titles & test_titles)
        self.assertEqual(len(validation) + len(test), len(self.rows))

    def test_fixed_split_sizes_and_reproducibility(self):
        config = QAConfig(mode="QUICK", quick_train_samples=12, quick_validation_samples=5, quick_test_samples=5)
        validation_rows = [{**row, "title": f"Dev {row['title']}"} for row in self.rows]
        dataset = {"train": self.rows, "validation": validation_rows}
        one = create_fixed_splits(dataset, config)
        two = create_fixed_splits(dataset, config)
        self.assertEqual({key: len(value) for key, value in one.items()}, {"train": 12, "validation": 5, "test": 5})
        self.assertEqual(one, two)
        self.assertFalse({r["title"] for r in one["validation"]} & {r["title"] for r in one["test"]})

    def test_split_title_validator_returns_sorted_titles(self):
        actual = validate_split_titles({"train": self.rows[:15], "validation": self.rows[15:]})
        self.assertEqual(actual["train"], sorted({row["title"] for row in self.rows[:15]}))

    def test_split_title_validator_rejects_leakage(self):
        with self.assertRaisesRegex(ValueError, "title leakage"):
            validate_split_titles({"train": [self.rows[0]], "validation": [self.rows[1]]})


class TestKeywordBaseline(unittest.TestCase):
    def test_answer_is_a_real_context_span(self):
        context = "Paris is the French capital. Lyon is another large city."
        result = keyword_overlap_baseline("What is the capital of France?", context)
        self.assertTrue(result["answer"])
        self.assertEqual(context[result["start_char"]:result["end_char"]], result["answer"])

    def test_sentence_offsets_are_preserved(self):
        context = "First sentence.  Second sentence!"
        for sentence, start, end in split_context_sentences(context):
            self.assertEqual(context[start:end], sentence)


class TestArtifactUtilities(unittest.TestCase):
    def test_results_csv_upsert_keeps_other_experiments_and_replaces_same_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            old = {"experiment": "ctx", "model": "m", "setting": "128", "split": "val", "seed": 42, "f1": 10}
            other = {"experiment": "baseline", "model": "b", "setting": "base", "split": "test", "seed": 42, "f1": 20}
            upsert_results_csv(path, [old, other])
            updated = {**old, "f1": 30}
            upsert_results_csv(path, [updated])
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            by_experiment = {row["experiment"]: row for row in rows}
            self.assertEqual(by_experiment["ctx"]["f1"], "30")
            self.assertEqual(by_experiment["baseline"]["f1"], "20")

    def test_environment_report_works_without_cuda(self):
        report = environment_report(QAConfig())
        self.assertIn("python", report)
        self.assertEqual(report["mode"], "QUICK")

    def test_checkpoint_resume_requires_saved_trainer_state(self):
        from qa_model import checkpoint_resume_path

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "run"
            checkpoint.mkdir()
            self.assertIsNone(checkpoint_resume_path(checkpoint, resume_existing=True))
            (checkpoint / "trainer_state.pt").write_bytes(b"state")
            self.assertEqual(checkpoint_resume_path(checkpoint, resume_existing=True), str(checkpoint))
            self.assertIsNone(checkpoint_resume_path(checkpoint, resume_existing=False))


class TestModelPlumbingSmoke(unittest.TestCase):
    def test_forward_qa_batch_drops_unsupported_token_type_ids(self):
        import torch
        from types import SimpleNamespace
        from qa_model import forward_qa_batch

        class StrictQA(torch.nn.Module):
            def forward(self, input_ids, attention_mask, start_positions, end_positions):
                self.received = {"input_ids", "attention_mask", "start_positions", "end_positions"}
                return SimpleNamespace(loss=input_ids.float().sum() * 0)

        model = StrictQA()
        batch = {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
            "start_positions": torch.tensor([1]),
            "end_positions": torch.tensor([1]),
        }
        result = forward_qa_batch(model, batch, torch.device("cpu"))
        self.assertTrue(torch.isfinite(result.loss).item())
        self.assertEqual(model.received, {"input_ids", "attention_mask", "start_positions", "end_positions"})

    def test_random_initialized_qa_head_shapes_and_loss(self):
        import torch
        from transformers import AutoModelForQuestionAnswering, DistilBertConfig
        config = DistilBertConfig(vocab_size=101, dim=32, hidden_dim=64, n_layers=1, n_heads=4)
        model = AutoModelForQuestionAnswering.from_config(config)
        inputs = torch.randint(0, 101, (2, 12))
        mask = torch.ones_like(inputs)
        output = model(input_ids=inputs, attention_mask=mask,
                       start_positions=torch.tensor([2, 3]), end_positions=torch.tensor([4, 6]))
        self.assertEqual(tuple(output.start_logits.shape), (2, 12))
        self.assertEqual(tuple(output.end_logits.shape), (2, 12))
        self.assertTrue(torch.isfinite(output.loss).item())

    def test_cpu_training_supports_gradient_accumulation_and_partial_final_group(self):
        import torch
        from types import SimpleNamespace
        from qa_model import train_one_epoch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids, attention_mask, start_positions, end_positions):
                target = start_positions.float().mean()
                loss = (self.weight - target).square()
                return SimpleNamespace(loss=loss)

        model = TinyModel()
        batches = [{"input_ids": torch.tensor([float(i)]), "attention_mask": torch.tensor([1]),
                    "start_positions": torch.tensor([1]), "end_positions": torch.tensor([1])}
                   for i in range(3)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = train_one_epoch(model, batches, optimizer, torch.device("cpu"),
                               mixed_precision=False, gradient_accumulation_steps=2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertGreater(float(model.weight.detach()), 0.0)

    def test_training_saves_best_validation_checkpoint_and_can_resume(self):
        import torch
        from types import SimpleNamespace
        from qa_model import QAConfig, train_with_validation

        class TinyQA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids, attention_mask, start_positions, end_positions):
                target = start_positions.float().mean()
                loss = (self.weight - target).square()
                return SimpleNamespace(loss=loss)

            def save_pretrained(self, directory):
                torch.save(self.state_dict(), Path(directory) / "pytorch_model.bin")

        batch = {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]),
                 "start_positions": torch.tensor([1]), "end_positions": torch.tensor([1])}
        train_loader = [batch, batch]
        val_loader = [batch]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QAConfig(epochs=2, mixed_precision=False, gradient_accumulation_steps=2)
            scores = iter([0.9, 0.2])
            model, history, _ = train_with_validation(
                TinyQA(), train_loader, val_loader,
                lambda _: {"em": 0.0, "f1": next(scores)}, cfg, tmp)
            saved = torch.load(Path(tmp) / "pytorch_model.bin", map_location="cpu", weights_only=True)
            self.assertEqual(len(history), 2)
            self.assertTrue(torch.equal(model.weight.detach().cpu(), saved["weight"]))

            cfg = QAConfig(epochs=3, mixed_precision=False)
            scores = iter([0.95])
            _, resumed_history, _ = train_with_validation(
                TinyQA(), train_loader, val_loader,
                lambda _: {"em": 0.0, "f1": next(scores)}, cfg, tmp, resume_from=tmp)
            self.assertEqual(len(resumed_history), 3)


class TestStudentTODOHarness(unittest.TestCase):
    def test_alignment_cases(self):
        from qa_model import character_to_token_span
        self.assertEqual(character_to_token_span([(10, 13), (14, 18)], [1, 1], 10, "red fox"), (0, 1))
        self.assertEqual(character_to_token_span([(10, 16)], [1], 12, "fox"), (0, 0))
        self.assertIsNone(character_to_token_span([(100, 104)], [1], 98, "abcdef"))

    def test_metric_cases(self):
        from qa_model import exact_match_score, token_f1_score
        self.assertEqual(exact_match_score("The Cat!", ["cat"]), 1.0)
        self.assertEqual(token_f1_score("red", ["blue"]), 0.0)
        self.assertEqual(token_f1_score("blue car", ["blue truck"]), 0.5)

    def test_window_coverage(self):
        from qa_model import window_answer_coverage
        example = {"answers": {"text": ["answer"], "answer_start": [102]}}
        complete = {"offset_mapping": [None, (100, 110)], "sequence_ids": [None, 1]}
        clipped = {"offset_mapping": [None, (104, 108)], "sequence_ids": [None, 1]}
        elsewhere = {"offset_mapping": [None, (10, 20)], "sequence_ids": [None, 1]}
        self.assertTrue(window_answer_coverage(example, complete))
        self.assertFalse(window_answer_coverage(example, clipped))
        self.assertFalse(window_answer_coverage(example, elsewhere))


if __name__ == "__main__":
    unittest.main(verbosity=2)
