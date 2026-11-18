from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

from baseline_ner import evaluate_baseline, predict_baseline, train_baseline
from information_extraction import decode_bio_words, group_entities
from ner_core import (
    IGNORE_INDEX, NerSplit, align_words, entity_counts, entity_length_metrics, load_local_conll, prediction_ready,
    extract_bio_spans, reconstruct_word_labels, validate_bio_sequences,
)


class FakeEncoding(dict):
    def __init__(self, values, word_ids):
        super().__init__(values)
        self._word_ids = word_ids
    def word_ids(self, batch_index=None):
        return list(self._word_ids)


class FakeFastTokenizer:
    is_fast = True
    def __call__(self, words, is_split_into_words=True, truncation=False, max_length=None, add_special_tokens=True, **kwargs):
        ids = [None] if add_special_tokens else []
        input_ids = [101] if add_special_tokens else []
        for index, word in enumerate(words):
            pieces = [11, 12] if word == "playing" else [13]
            for piece in pieces:
                ids.append(index); input_ids.append(piece)
        if add_special_tokens:
            ids.append(None); input_ids.append(102)
        if truncation and max_length is not None:
            ids = ids[:max_length]; input_ids = input_ids[:max_length]
        return FakeEncoding({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}, ids)


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeFastTokenizer()
        self.label2id = {"O": 0, "B-PER": 1, "I-PER": 2}

    def test_unsplit_word(self):
        result = align_words(["John"], ["B-PER"], self.tokenizer, self.label2id, 10)
        self.assertEqual(result["aligned_labels"], [-100, 1, -100])
        self.assertEqual(result["kept_word_ids"], [0])

    def test_split_word_first_piece_only(self):
        result = align_words(["playing"], ["B-PER"], self.tokenizer, self.label2id, 10)
        self.assertEqual(result["aligned_labels"], [-100, 1, -100, -100])
        self.assertEqual(result["split_words"], {0: 2})

    def test_special_token_padding_and_truncation(self):
        full = align_words(["John", "playing"], ["B-PER", "I-PER"], self.tokenizer, self.label2id, 10)
        self.assertEqual(full["word_ids"][0], None)
        truncated = align_words(["John", "playing"], ["B-PER", "I-PER"], self.tokenizer, self.label2id, 2)
        self.assertEqual(truncated["kept_word_ids"], [0])
        self.assertEqual(truncated["truncated_word_ids"], [1])
        # A data collator pads labels with -100; confirm that padding is not a class ID.
        padded = full["aligned_labels"] + [IGNORE_INDEX]
        self.assertEqual(padded[-1], -100)

    def test_word_prediction_reconstruction(self):
        indices, predictions = reconstruct_word_labels([None, 0, 0, 1, None], [0, 1, 2, 0, 0], ["O", "B-PER", "I-PER"], 2)
        self.assertEqual(indices, [0, 1])
        self.assertEqual(predictions, [1, 0])

    def test_requires_fast_tokenizer(self):
        self.tokenizer.is_fast = False
        with self.assertRaises(ValueError):
            align_words(["John"], ["B-PER"], self.tokenizer, self.label2id)


class TestDatasetLoading(unittest.TestCase):
    def test_local_conll_split_loader_preserves_words_and_tags(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for filename in ["train.txt","validation.txt","test.txt"]:
                (root/filename).write_text("John NNP B-NP B-PER\nworks VBZ B-VP O\n\n",encoding="utf-8")
            splits,names=load_local_conll(root)
            self.assertEqual(set(splits),{"train","validation","test"})
            self.assertEqual(splits["train"].words,[["John","works"]])
            self.assertEqual(splits["train"].labels,[["B-PER","O"]])
            self.assertIn("B-PER",names)

    def test_student_prediction_gate(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/"prediction.md"
            path.write_text("TODO: make predictions",encoding="utf-8")
            self.assertFalse(prediction_ready(path)[0])
            path.write_text("Prediction 1: model X. Prediction 2: entity Y. Prediction 3: rare Z.",encoding="utf-8")
            self.assertTrue(prediction_ready(path)[0])


class TestEntityMetrics(unittest.TestCase):
    def test_span_boundaries_and_exact_match(self):
        self.assertEqual(extract_bio_spans(["B-LOC", "I-LOC", "O"]), {("LOC", 0, 2)})
        score = entity_counts([["B-LOC", "I-LOC"]], [["B-LOC", "O"]], ["LOC"])
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (0, 1, 1))

    def test_invalid_bio_is_reported(self):
        errors = validate_bio_sequences([["I-PER", "O"]])
        self.assertEqual(len(errors), 1)
        with self.assertRaises(ValueError):
            extract_bio_spans(["I-PER"])

    def test_entity_length_buckets(self):
        rows = entity_length_metrics([["B-PER", "I-PER", "O", "B-LOC"]], [["B-PER", "I-PER", "O", "O"]])
        self.assertEqual(rows[1]["tp"], 1)
        self.assertEqual(rows[0]["fn"], 1)


class TestBaselineAndIE(unittest.TestCase):
    def test_baseline_preserves_sentence_order_and_token_count(self):
        train = NerSplit(
            words=[["John", "works"], ["Paris", "grows"]] * 3,
            labels=[["B-PER", "O"], ["B-LOC", "O"]] * 3,
        )
        model, vectorizer, _ = train_baseline(train, seed=42)
        pred = predict_baseline(model, vectorizer, train)
        self.assertEqual([len(x) for x in pred], [2] * 6)
        metrics = evaluate_baseline(train, pred, ["PER", "LOC"])
        self.assertIn("entity_f1_micro", metrics)

    def test_information_extraction_decoding_and_json(self):
        entities = decode_bio_words(["Barack", "Obama", "visited", "Paris"], ["B-PER", "I-PER", "O", "B-LOC"])
        output = group_entities(entities)
        self.assertEqual(output, {"PERSON": ["Barack Obama"], "LOCATION": ["Paris"]})
        json.dumps(output)

    def test_ie_no_entity_and_repeated_entity(self):
        self.assertEqual(decode_bio_words(["Nothing", "named"], ["O", "O"]), [])
        entities = decode_bio_words(["Paris", "and", "Paris"], ["B-LOC", "O", "B-LOC"])
        self.assertEqual(group_entities(entities), {"LOCATION": ["Paris", "Paris"]})

    def test_ie_surface_and_character_offsets(self):
        import torch
        from information_extraction import predict_entities

        class TextEncoding(dict):
            def __init__(self, values, word_ids):
                super().__init__(values); self._word_ids = word_ids
            def word_ids(self): return list(self._word_ids)

        class TextTokenizer:
            def __call__(self, text, **kwargs):
                values = {
                    "input_ids": torch.tensor([[101, 10, 11, 12, 13, 14, 102]]),
                    "attention_mask": torch.ones((1, 7), dtype=torch.long),
                    "offset_mapping": torch.tensor([[[0,0],[0,6],[7,12],[13,20],[21,26],[26,27],[0,0]]]),
                }
                return TextEncoding(values, [None, 0, 1, 2, 3, 4, None])

        class FakeModel:
            config = SimpleNamespace(id2label={0:"O",1:"B-PER",2:"I-PER",3:"B-LOC"})
            def eval(self): return self
            def __call__(self, **kwargs):
                tags = [0,1,2,0,3,0,0]
                logits = torch.full((1,7,4), -5.0)
                for i, tag in enumerate(tags): logits[0,i,tag] = 5.0
                return SimpleNamespace(logits=logits)

        result = predict_entities("Barack Obama visited Paris.", FakeModel(), TextTokenizer(), torch.device("cpu"))
        self.assertEqual(result["structured"], {"PERSON":["Barack Obama"], "LOCATION":["Paris"]})
        self.assertEqual([(e["start"],e["end"]) for e in result["entities"]], [(0,12),(21,26)])
        json.dumps(result, ensure_ascii=False)

    def test_custom_entity_f1_matches_seqeval(self):
        from seqeval.metrics import f1_score
        gold = [["B-PER","I-PER","O"],["O","B-LOC"]]
        pred = [["B-PER","O","O"],["O","B-LOC"]]
        custom = entity_counts(gold,pred,["PER","LOC"])["f1"]
        self.assertAlmostEqual(custom, f1_score(gold,pred,scheme=None,zero_division=0), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
