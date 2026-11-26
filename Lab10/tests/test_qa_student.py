"""Unit tests for QA span alignment, feature preparation, decoding and scoring."""

from __future__ import annotations

import re
import unittest


class FakeEncoding(dict):
    def __init__(self, *args, sequence_id_rows, **kwargs):
        super().__init__(*args, **kwargs)
        self._sequence_id_rows = sequence_id_rows

    def sequence_ids(self, batch_index=0):
        return self._sequence_id_rows[batch_index]


class FakeFastTokenizer:
    """Small deterministic pair tokenizer for testing windows without model downloads."""

    is_fast = True
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    cls_token_id = 101
    sep_token_id = 102

    @staticmethod
    def _words(text):
        return [(match.group(), match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def __call__(self, questions, contexts, truncation, max_length, stride,
                 return_overflowing_tokens, return_offsets_mapping, padding=False):
        if isinstance(questions, str):
            questions, contexts = [questions], [contexts]
        rows = {"input_ids": [], "attention_mask": [], "token_type_ids": [],
                "offset_mapping": [], "overflow_to_sample_mapping": []}
        sequence_rows = []
        for sample_index, (question, context) in enumerate(zip(questions, contexts)):
            q_words = self._words(question)
            c_words = self._words(context)
            capacity = max_length - len(q_words) - 3
            if capacity <= 0 or stride >= capacity:
                raise ValueError("fake tokenizer has no room for context tokens at this max_length/stride")
            step = capacity - stride
            starts = list(range(0, max(len(c_words), 1), step))
            if not c_words:
                starts = [0]
            for c_start in starts:
                selected = c_words[c_start:c_start + capacity]
                ids = [self.cls_token_id]
                offsets = [(0, 0)]
                seq_ids = [None]
                types = [0]
                for _, start, end in q_words:
                    ids.append(10); offsets.append((start, end)); seq_ids.append(0); types.append(0)
                ids.append(self.sep_token_id); offsets.append((0, 0)); seq_ids.append(None); types.append(0)
                for _, start, end in selected:
                    ids.append(20 + len(ids)); offsets.append((start, end)); seq_ids.append(1); types.append(1)
                ids.append(self.sep_token_id); offsets.append((0, 0)); seq_ids.append(None); types.append(1)
                rows["input_ids"].append(ids)
                rows["attention_mask"].append([1] * len(ids))
                rows["token_type_ids"].append(types)
                rows["offset_mapping"].append(offsets)
                rows["overflow_to_sample_mapping"].append(sample_index)
                sequence_rows.append(seq_ids)
                if c_start + capacity >= len(c_words):
                    break
        return FakeEncoding(rows, sequence_id_rows=sequence_rows)


class TestCharacterTokenAlignmentStudentCases(unittest.TestCase):
    def setUp(self):
        from qa_model import character_to_token_span
        self.align = character_to_token_span

    def test_single_token_answer(self):
        self.assertEqual(self.align([(0, 0), (10, 15), (0, 0)], [None, 1, None], 11, "abc"), (1, 1))

    def test_multi_token_answer(self):
        offsets = [(0, 0), (10, 13), (14, 18), (0, 0)]
        self.assertEqual(self.align(offsets, [None, 1, 1, None], 10, "red fox"), (1, 2))

    def test_answer_start_inside_token(self):
        self.assertEqual(self.align([(0, 0), (10, 16), (0, 0)], [None, 1, None], 12, "fox"), (1, 1))

    def test_answer_end_inside_token(self):
        self.assertEqual(self.align([(0, 0), (10, 16), (0, 0)], [None, 1, None], 10, "fox"), (1, 1))

    def test_answer_outside_current_window(self):
        offsets, sequence_ids = [(0, 0), (100, 104), (105, 109), (0, 0)], [None, 1, 1, None]
        self.assertIsNone(self.align(offsets, sequence_ids, 20, "cat"))
        self.assertIsNone(self.align(offsets, sequence_ids, 120, "cat"))

    def test_answer_truncated(self):
        offsets, sequence_ids = [(0, 0), (100, 104), (105, 109), (110, 114), (0, 0)], [None, 1, 1, 1, None]
        self.assertIsNone(self.align(offsets, sequence_ids, 98, "abcdef"))
        self.assertIsNone(self.align(offsets, sequence_ids, 112, "abcdef"))

    def test_answer_fully_inside_window(self):
        offsets = [(0, 0), (100, 104), (105, 109), (110, 114), (0, 0)]
        self.assertEqual(self.align(offsets, [None, 1, 1, 1, None], 105, "wxyz"), (2, 2))

    def test_repeated_answer_text_uses_annotated_character_start(self):
        offsets = [(10, 13), (14, 17), (30, 33), (34, 37)]
        self.assertEqual(self.align(offsets, [1, 1, 1, 1], 30, "red"), (2, 2))

    def test_multiple_windows_keep_original_sample_id(self):
        from qa_model import prepare_validation_features
        tokenizer = FakeFastTokenizer()
        examples = [{"id": "doc-1", "question": "which word", "context": "a b c d e f g h i"}]
        features = prepare_validation_features(tokenizer, examples, max_length=8, stride=1)
        self.assertGreater(len(features), 1)
        self.assertEqual({feature["overflow_to_sample_mapping"] for feature in features}, {0})
        self.assertEqual({feature["example_id"] for feature in features}, {"doc-1"})

    def test_character_end_is_exclusive(self):
        offsets = [(0, 0), (4, 8), (8, 11)]
        self.assertEqual(self.align(offsets, [None, 1, 1], 4, "York"), (1, 1))

    def test_question_tokens_are_never_answer_tokens(self):
        offsets = [(1, 4), (0, 0), (10, 13)]
        self.assertIsNone(self.align(offsets, [0, None, 1], 1, "cat"))

    def test_special_and_padding_tokens_are_excluded(self):
        offsets = [(0, 0), (0, 0), (10, 13), (0, 0)]
        self.assertEqual(self.align(offsets, [None, None, 1, None], 10, "cat"), (2, 2))


class TestFeaturePreparationStudentCases(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeFastTokenizer()
        self.example = {"id": "ex-1", "question": "which word", "context": "alpha beta gamma delta epsilon zeta",
                        "answers": {"text": ["delta epsilon"], "answer_start": [17]}}

    def test_training_features_keep_window_labels_and_source_mapping(self):
        from qa_model import prepare_train_features, window_answer_coverage
        features = prepare_train_features(self.tokenizer, [self.example], max_length=9, stride=1)
        self.assertEqual(len(features), 2)
        self.assertEqual([feature["overflow_to_sample_mapping"] for feature in features], [0, 0])
        self.assertEqual((features[0]["start_positions"], features[0]["end_positions"]), (0, 0))
        # second feature contains "delta epsilon" at token indices 4 and 5
        self.assertEqual((features[1]["start_positions"], features[1]["end_positions"]), (4, 5))
        self.assertTrue(all(len(feature["offset_mapping"]) == len(feature["sequence_ids"])
                            for feature in features))
        self.assertEqual(sum(window_answer_coverage(self.example, feature) for feature in features), 1)

    def test_validation_features_keep_ids_offsets_and_sequence_mapping(self):
        from qa_model import prepare_validation_features
        features = prepare_validation_features(self.tokenizer, [self.example], max_length=9, stride=1)
        self.assertEqual(len(features), 2)
        self.assertTrue(all(feature["example_id"] == "ex-1" for feature in features))
        self.assertTrue(all(feature["overflow_to_sample_mapping"] == 0 for feature in features))
        for feature in features:
            self.assertEqual(len(feature["offset_mapping"]), len(feature["sequence_ids"]))
            self.assertTrue(all(offset is None for offset, sid in zip(feature["offset_mapping"], feature["sequence_ids"]) if sid != 1))
        self.assertEqual((features[0]["start_positions"], features[0]["end_positions"]), (0, 0))
        self.assertEqual((features[1]["start_positions"], features[1]["end_positions"]), (4, 5))
        self.assertNotIn("answers", features[0])

    def test_validation_windows_without_answer_use_cls_labels(self):
        from qa_model import prepare_validation_features
        example = {**self.example, "answers": {"text": [], "answer_start": []}}
        features = prepare_validation_features(self.tokenizer, [example], max_length=9, stride=1)
        self.assertTrue(features)
        self.assertTrue(all((feature["start_positions"], feature["end_positions"]) == (0, 0)
                            for feature in features))

    def test_answer_coverage_counts_a_complete_span_once(self):
        from qa_model import prepare_validation_features, window_answer_coverage
        features = prepare_validation_features(self.tokenizer, [self.example], max_length=9, stride=1)
        self.assertEqual(sum(window_answer_coverage(self.example, feature) for feature in features), 1)


class TestSpanDecoderStudentCases(unittest.TestCase):
    def test_decoder_excludes_question_special_and_padding(self):
        from qa_model import decode_qa_predictions
        example = {"id": "e1", "context": "Paris is great", "question": "where"}
        feature = {"example_id": "e1", "offset_mapping": [None, (0, 5), (6, 8), (9, 14), None],
                   "sequence_ids": [None, 0, 1, 1, None]}
        # Highest invalid logits are ignored; the best valid context token is "great".
        starts = [[100, 90, 2, 8, 80]]
        ends = [[100, 90, 2, 8, 80]]
        result = decode_qa_predictions([example], [feature], starts, ends, max_answer_length=2)[0]
        self.assertEqual(result["prediction"], "great")
        self.assertEqual((result["start_char"], result["end_char"]), (9, 14))

    def test_decoder_rejects_reversed_and_overlong_spans(self):
        from qa_model import decode_qa_predictions
        example = {"id": "e1", "context": "red fox runs", "question": "what"}
        feature = {"example_id": "e1", "offset_mapping": [None, (0, 3), (4, 7), (8, 12), None],
                   "sequence_ids": [None, 1, 1, 1, None]}
        starts = [[0, 2, 10, 1, 0]]  # strongest start is at "fox"
        ends = [[0, 10, 3, 1, 0]]    # strongest end is before that start
        result = decode_qa_predictions([example], [feature], starts, ends, max_answer_length=1)[0]
        self.assertEqual(result["prediction"], "fox")
        self.assertEqual((result["start_char"], result["end_char"]), (4, 7))

    def test_decoder_aggregates_windows_and_returns_original_offsets(self):
        from qa_model import decode_qa_predictions
        example = {"id": "e1", "context": "alpha beta gamma", "question": "which?"}
        features = [
            {"overflow_to_sample_mapping": 0, "offset_mapping": [None, (0, 5), (6, 10)], "sequence_ids": [None, 1, 1]},
            {"overflow_to_sample_mapping": 0, "offset_mapping": [None, (6, 10), (11, 16)], "sequence_ids": [None, 1, 1]},
        ]
        starts = [[0, 5, 1], [0, 12, 1]]
        ends = [[0, 5, 1], [0, 12, 1]]
        result = decode_qa_predictions([example], features, starts, ends)[0]
        self.assertEqual(result["prediction"], "beta")
        self.assertEqual((result["start_char"], result["end_char"]), (6, 10))


class TestQAMetricsStudentCases(unittest.TestCase):
    def test_exact_match(self):
        from qa_model import exact_match_score
        self.assertEqual(exact_match_score("The Cat.", ["cat"]), 1.0)

    def test_no_token_overlap(self):
        from qa_model import token_f1_score
        self.assertEqual(token_f1_score("red", ["blue"]), 0.0)

    def test_partial_overlap(self):
        from qa_model import token_f1_score
        self.assertEqual(token_f1_score("blue car", ["blue truck"]), 0.5)

    def test_case_and_punctuation_normalization(self):
        from qa_model import exact_match_score
        self.assertEqual(exact_match_score("NEW YORK!", ["New York"]), 1.0)

    def test_repeated_tokens(self):
        from qa_model import token_f1_score
        self.assertAlmostEqual(token_f1_score("cat cat dog", ["cat dog dog"]), 2 / 3)

    def test_multiple_gold_answers(self):
        from qa_model import exact_match_score, token_f1_score
        self.assertEqual(exact_match_score("NYC", ["New York", "NYC"]), 1.0)
        self.assertEqual(token_f1_score("NYC", ["New York", "NYC"]), 1.0)

    def test_empty_prediction(self):
        from qa_model import score_one_example
        self.assertEqual(score_one_example("", ["answer"]), {"em": 0.0, "f1": 0.0})
        self.assertEqual(score_one_example("", []), {"em": 100.0, "f1": 100.0})

    def test_normalization_matches_squad_rules(self):
        from qa_model import normalize_answer
        self.assertEqual(normalize_answer(" The, QUICK! brown fox's  " ), "quick brown foxs")

    def test_token_f1_repeated_token_counts(self):
        from qa_model import token_f1_score
        self.assertAlmostEqual(token_f1_score("cat cat dog", ["cat dog dog"]), 2 / 3)
        self.assertAlmostEqual(token_f1_score("cat cat", ["cat"]), 2 / 3)

    def test_dataset_aggregation_and_multiple_references(self):
        from qa_model import evaluate_predictions
        predictions = [{"id": "b", "prediction": "NYC"}]
        references = [
            {"id": "a", "answers": {"text": ["red"]}},
            {"id": "b", "answers": {"text": ["New York", "NYC"]}},
        ]
        result = evaluate_predictions(predictions, references)
        self.assertEqual(result, {"em": 50.0, "f1": 50.0})


class TestSpanErrorAnalysisStudentCases(unittest.TestCase):
    def test_error_categories_use_original_character_coordinates(self):
        from qa_model import classify_span_errors
        context = "New York. Boston."
        gold = {"text": ["New York", "Boston"], "answer_start": [0, 10]}
        cases = [
            ({"start_char": 0, "end_char": 8}, "correct", False, False),
            ({"start_char": 1, "end_char": 8}, "wrong_start_only", True, False),
            ({"start_char": 0, "end_char": 7}, "wrong_end_only", False, True),
            ({"start_char": 1, "end_char": 7}, "both_boundaries_wrong", True, True),
            ({"start_char": 8, "end_char": 10}, "non_overlapping", True, True),
            ({"start_char": 0, "end_char": 7}, "wrong_end_only", False, True),
            ({"start_char": 10, "end_char": 16}, "correct", False, False),
        ]
        for prediction, expected_type, wrong_start, wrong_end in cases:
            with self.subTest(prediction=prediction):
                result = classify_span_errors("where?", context, gold, prediction)
                self.assertEqual(result["error_type"], expected_type)
                self.assertEqual(result["wrong_start"], wrong_start)
                self.assertEqual(result["wrong_end"], wrong_end)
                self.assertEqual(result["wrong_span"], expected_type != "correct")
        partial = classify_span_errors("where?", context, {"text": ["New York"], "answer_start": [0]},
                                       {"start_char": 1, "end_char": 7})
        self.assertEqual(partial["overlap_type"], "partial")
        self.assertEqual(partial["error_type"], "both_boundaries_wrong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
