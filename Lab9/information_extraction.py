"""Turn a fine-tuned CoNLL-2003 token classifier output into structured entities."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ENTITY_KEY_MAP = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISCELLANEOUS"}


def decode_bio_words(words: list[str], tags: list[str]) -> list[dict[str, Any]]:
    if len(words) != len(tags):
        raise ValueError("word/tag length mismatch")
    entities = []
    start = None; entity_type = None
    for index, tag in enumerate(list(tags) + ["O"]):
        prefix, current_type = ("O", None) if tag == "O" else tag.split("-", 1)
        continuation = prefix == "I" and start is not None and current_type == entity_type
        if prefix == "I" and not continuation:
            # The model may emit an invalid IOB2 transition; retain evidence as a new span.
            prefix = "B"
        if start is not None and not (prefix == "I" and continuation):
            entities.append({"text": " ".join(words[start:index]), "label": entity_type,
                             "output_key": ENTITY_KEY_MAP.get(entity_type, entity_type),
                             "start_word": start, "end_word": index})
            start = None; entity_type = None
        if prefix == "B":
            start = index; entity_type = current_type
    return entities


def group_entities(entities: list[dict[str, Any]], include_metadata: bool = False) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for entity in entities:
        key = entity["output_key"]
        value: Any = entity
        if not include_metadata:
            value = entity["text"]
        grouped[key].append(value)
    return dict(grouped)


def predict_entities(text: str, model: Any, tokenizer: Any, device: Any, max_length: int = 256) -> dict[str, Any]:
    import torch
    if not text.strip():
        return {"text": text, "entities": [], "structured": {}}
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
    ids = logits.argmax(-1).detach().cpu().tolist()
    id2label = model.config.id2label
    word_ids = tokenizer(text, truncation=True, max_length=max_length).word_ids()
    words: list[dict[str, Any]] = []
    seen = set()
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        token_offsets = [offsets[j] for j, wid in enumerate(word_ids) if wid == word_id and offsets[j][1] > offsets[j][0]]
        if not token_offsets:
            continue
        left = min(pair[0] for pair in token_offsets); right = max(pair[1] for pair in token_offsets)
        words.append({"word_id": word_id, "text": text[left:right], "start": left, "end": right, "tag": id2label[int(ids[token_index])]})
    word_tags = [item["tag"] for item in words]
    word_surfaces = [item["text"] for item in words]
    entities = decode_bio_words(word_surfaces, word_tags)
    # Add source character offsets while preserving the exact surface in input text.
    for entity in entities:
        left_word, right_word = entity.pop("start_word"), entity.pop("end_word")
        entity["start"] = words[left_word]["start"]
        entity["end"] = words[right_word - 1]["end"]
        entity["text"] = text[entity["start"]:entity["end"]]
    return {"text": text, "entities": entities, "structured": group_entities(entities),
            "truncated": bool(encoded["input_ids"].shape[1] >= max_length)}


def load_extractor(checkpoint: str):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    location = Path(checkpoint)
    if not location.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {location}. Run ner_finetuning.ipynb first.")
    tokenizer = AutoTokenizer.from_pretrained(str(location), use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("The NER inference path requires a fast tokenizer")
    model = AutoModelForTokenClassification.from_pretrained(str(location))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--metadata", action="store_true", help="Include label and character offsets in grouped values")
    args = parser.parse_args()
    model, tokenizer, device = load_extractor(args.checkpoint)
    result = predict_entities(args.text, model, tokenizer, device, args.max_length)
    if args.metadata:
        result["structured"] = group_entities(result["entities"], include_metadata=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
