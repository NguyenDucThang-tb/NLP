"""CLI scaffold for extractive QA over a supplied document/context (no retrieval)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from qa_model import build_tokenizer, decode_qa_predictions


DEMO_CONTEXT = (
    "The Northstar Museum opened in 1988 in Riverton. Its first director was Elena Park. "
    "The museum's archive contains maps, photographs, and letters from the region. "
    "A renovation completed in 2016 added a second exhibition hall."
)
DEMO_QUESTIONS = [
    "When did the Northstar Museum open?",
    "Who was its first director?",
    "What did the renovation add?",
]


def predict_span(question: str, context: str, model: Any, tokenizer: Any,
                 max_length: int = 384, stride: int = 128, max_answer_length: int = 30) -> dict[str, Any]:
    """Run model logits on context windows and delegate span choice to student code."""
    if not question.strip() or not context.strip():
        raise ValueError("Question and context must be non-empty")
    if max_length < 16 or stride < 0 or stride >= max_length:
        raise ValueError("Require max_length >= 16 and 0 <= stride < max_length")
    encoded = tokenizer(
        question, context,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    features = []
    for feature_id, input_ids in enumerate(encoded["input_ids"]):
        row = {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"][feature_id],
            "offset_mapping": encoded["offset_mapping"][feature_id],
            "sequence_ids": encoded.sequence_ids(feature_id),
            "overflow_to_sample_mapping": encoded["overflow_to_sample_mapping"][feature_id],
            "example_id": "application-input",
        }
        if "token_type_ids" in encoded:
            row["token_type_ids"] = encoded["token_type_ids"][feature_id]
        features.append(row)

    device = next(model.parameters()).device
    model.eval()
    start_logits, end_logits = [], []
    with torch.no_grad():
        for feature in features:
            inputs = {key: torch.tensor([feature[key]], dtype=torch.long, device=device)
                      for key in tokenizer.model_input_names if key in feature}
            outputs = model(**inputs)
            start_logits.append(outputs.start_logits[0].cpu().numpy())
            end_logits.append(outputs.end_logits[0].cpu().numpy())

    example = {"id": "application-input", "question": question, "context": context,
               "answers": {"text": [], "answer_start": []}}
    prediction = decode_qa_predictions([example], features, start_logits, end_logits, max_answer_length)
    return prediction[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Student fine-tuned QA checkpoint directory")
    parser.add_argument("--question")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--context", help="Context passed directly on the command line")
    source.add_argument("--document", type=Path, help="UTF-8 text file used as context")
    parser.add_argument("--demo", action="store_true", help="Run three questions on the built-in sample document")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=30)
    args = parser.parse_args()

    if not args.demo and (not args.question or not (args.context or args.document)):
        parser.error("provide --question with --context/--document, or use --demo")
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {model_path}")

    from transformers import AutoModelForQuestionAnswering
    tokenizer = build_tokenizer(str(model_path))
    model = AutoModelForQuestionAnswering.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    context = DEMO_CONTEXT if args.demo else (args.context if args.context is not None else args.document.read_text(encoding="utf-8"))
    questions = DEMO_QUESTIONS if args.demo else [args.question]

    for question in questions:
        result = predict_span(question, context, model, tokenizer, args.max_length,
                              args.stride, args.max_answer_length)
        start, end = result.get("start_char"), result.get("end_char")
        print({"question": question, "answer": result.get("prediction", result.get("answer")),
               "start_char": start, "end_char": end,
               "context_excerpt": context[max(0, (start or 0) - 80):min(len(context), (end or 0) + 80)]})


if __name__ == "__main__":
    main()

