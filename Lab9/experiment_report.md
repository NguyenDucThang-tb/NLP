# LAB 09 — Experiment report

## Run status

Local verification was run on 18 Nov 2026. This machine is CPU-only (`torch.cuda.is_available() == False`; `nvidia-smi` is unavailable). The student has now completed `prediction.md`. The baseline was rerun afterward and is marked `EXECUTED` in `results.csv`; its latest measured fit time is 20.5719 seconds. An earlier baseline run, before prediction was completed, was exploratory and is superseded. The saved tokenization notebook output is still from the earlier exploratory run; refreshing it through Jupyter was blocked here because the sandbox does not permit opening the kernel's local sockets. Official Transformer fine-tuning was not run because no CUDA device is available, and no Transformer test metrics are reported.

Dataset source: `tner/conll2003` (CoNLL-2003). Datasets 4.x no longer executes the legacy Hub dataset script; the loader uses JSONL and `label.json` from the same repository, without substituting another dataset. Label order came from metadata.

| Split | Sentences | Words | Invalid gold BIO sequences |
|---|---:|---:|---:|
| Train | 14,041 | 203,618 | 0 |
| Validation | 3,250 | 51,360 | 0 |
| Test | 3,453 | 46,427 | 0 |

Environment versions and hardware are recorded in `outputs/environment.json`.

## Observed results

### Experiment 1 — Tokenization sample

The notebook executed on 20 training sentences: 10 shorter sentences (5 words each) and 10 longer sentences (58–113 words). Special tokens are excluded from counts.

| Sample statistic | Value |
|---|---:|
| Mean R across 20 sentences | 1.9154 |
| Median R | 1.5566 |
| Maximum R | 4.2000 |
| Mean R, short group | 2.4800 |
| Mean R, long group | 1.3507 |
| Entity-word split ratio | 0.3274 |
| Non-entity-word split ratio | 0.2049 |

Per-sentence data and plots are saved under `outputs/tokenization/`.

### Baseline — token features + Logistic Regression

Fit used the train split; evaluation used the original 3,453-sentence test split (46,427 words). Measured classifier fit time was 22.729 seconds. DictVectorizer and classifier were fitted on train only.

| Metric | Observed value |
|---|---:|
| Token precision (micro) | 0.9531 |
| Token recall (micro) | 0.9531 |
| Token F1 (micro) | 0.9531 |
| Token accuracy | 0.9531 |
| Strict entity precision (micro) | 0.6980 |
| Strict entity recall (micro) | 0.7433 |
| Strict entity F1 (micro) | 0.7199 |
| Entity TP / FP / FN | 4198 / 1816 / 1450 |
| Invalid predicted BIO transitions | 590 |
| Invalid gold BIO transitions | 0 |
| Baseline feature count | 99389 |

Custom strict exact-span precision/recall/F1 match `seqeval` on these same baseline predictions. Per-type results:

| entity_type | precision | recall | f1 | support | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| LOC | 0.7811 | 0.8106 | 0.7955 | 1668 | 1352 | 379 | 316 |
| MISC | 0.7228 | 0.7578 | 0.7399 | 702 | 532 | 204 | 170 |
| ORG | 0.5916 | 0.6123 | 0.6018 | 1661 | 1017 | 702 | 644 |
| PER | 0.7095 | 0.8021 | 0.7530 | 1617 | 1297 | 531 | 320 |

Entity length results use original word lengths:

| entity_length | gold_count | predicted_count | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3574 | 4271 | 2733 | 1538 | 841 | 0.6399 | 0.7647 | 0.6967 |
| 2 | 1776 | 1595 | 1361 | 234 | 415 | 0.8533 | 0.7663 | 0.8075 |
| 3 | 236 | 129 | 92 | 37 | 144 | 0.7132 | 0.3898 | 0.5041 |
| 4+ | 62 | 19 | 12 | 7 | 50 | 0.6316 | 0.1935 | 0.2963 |

`confusion_matrix.png` is a **token-level BIO confusion matrix**. `error_analysis.md` contains baseline candidates only; explanations remain TODO.

### Context and Information Extraction artifacts

`context_analysis.md` contains 69 same-surface/different-gold-type rows from the test split. `ie_gold_template.csv` contains 30 test sentences with spans from CoNLL gold annotations. IE inference/evaluation is NOT_RUN because there is no fine-tuned checkpoint; `ie_evaluation.csv` records status without scores.

## Verification

- `python -m unittest -v test_lab09.py`: 15 tests passed, including alignment, ignore index, truncation, reconstruction, baseline output lengths, exact-span scoring, custom-vs-seqeval parity and IE offsets/JSON.
- `tokenizer_alignment.ipynb`: executed end-to-end with outputs saved.
- `ner_finetuning.ipynb`: executed through data/alignment/artifact cells; training and Transformer test cells reported NOT_RUN due prediction gate/no checkpoint.
- CPU forward smoke: logits `(2, 12, 9)`, finite loss, randomly initialized model. This checks model plumbing only; it is not pretrained inference or a training result.

## Status by component

| Component | Status | Evidence / condition |
|---|---|---|
| Dataset inspection and label mapping | EXECUTED | CoNLL-2003 splits and metadata labels; zero invalid gold BIO |
| Experiment 1 tokenization | EXECUTED_EXPLORATORY | 20 real sentences, tables and plots; saved notebook output predates completed prediction; rerun notebook now |
| Baseline test evaluation | EXECUTED | Latest run after completed prediction; `results.csv`, model and test predictions updated |
| Entity-level evaluator | TESTED | 15 tests; custom F1 equals seqeval on baseline predictions |
| Transformer training/comparison | NOT_RUN | No local CUDA; no fine-tuned checkpoint |
| Transformer per-type/length/error analysis | NOT_RUN | Requires fine-tuned checkpoint and test predictions |
| Context analysis | EXECUTED | Real test occurrences in `context_analysis.md` |
| IE code and 30-sentence gold sample | IMPLEMENTED / TESTED | Decoder/offset tests pass; source is CoNLL annotations |
| IE inference/evaluation | NOT_RUN | Requires fine-tuned checkpoint |

## Student interpretation — TODO

Tự phân tích token-level và exact entity-level khác nhau thế nào; đối chiếu entity types và lengths bằng các bảng; chọn context/error examples sau khi kiểm tra câu gốc; và giải thích giới hạn của baseline. Không suy ra nguyên nhân chỉ từ metrics hoặc nhãn candidate.
