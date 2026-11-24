# LAB 10 — Extractive Question Answering

Lab 10 biến đầu ra Transformer thành hai vị trí trên context: answer start và answer end. Model trích một span đã có trong văn bản; phần application không truy xuất tài liệu và không sinh câu mới.

## Repository structure

```text
Lab10/
├── W10.pdf
├── README.md
├── calculations.md
├── prediction.md
├── tokenizer_alignment.ipynb
├── qa_model.py
├── student_implementation_guide.md
├── qa_finetuning.ipynb
├── requirements.txt
├── tests/test_qa_framework.py
├── results.csv                 # được tạo sau khi có experiment thật
├── learning_curves.png         # được tạo sau khi training thật
├── em_f1_analysis.png          # được tạo sau evaluation thật
├── answer_length_analysis.png  # được tạo sau evaluation thật
├── error_analysis.md
├── document_qa.py
└── reflection.md
```

`results.csv` và các PNG chưa được tạo sẵn để tránh nhầm template với số liệu chạy thật. Checkpoint và dataset cache không nên commit.

## Dataset and split policy

Mặc định dùng SQuAD v1.1 (`rajpurkar/squad`), gồm `question`, `context`, `answers.text`, `answers.answer_start`. Dataset công khai có train và validation, không có test có nhãn công khai theo cấu hình chuẩn. Vì thế code lấy train từ split `train`, chia split `validation` gốc theo **article title** thành validation và held-out evaluation/test; một title không xuất hiện ở cả hai bên. Seed là 42. QUICK/FULL chỉ giới hạn số lượng trong các split cố định này, không tạo lại split.

FULL mặc định tối đa 10.000 train, 1.000 validation và 1.000 test examples. QUICK mặc định 64/16/16 để kiểm tra luồng code sau khi sinh viên hoàn tất alignment và metrics.

## Environment setup

### Local Linux/Windows

Tạo môi trường riêng. Với PyTorch, cài bản phù hợp phần cứng trước; `requirements.txt` không pin bản CPU/CUDA để không ghi đè PyTorch GPU của Colab.

```bash
cd /home/nguyenducthang/NLP/Lab10
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Cài PyTorch phù hợp CPU/GPU theo hướng dẫn chính thức của PyTorch.
```

### Google Colab T4

1. Push repository lên GitHub; mở notebook trong Colab và chọn **Runtime → Change runtime type → T4 GPU**.
2. Chạy cell setup đầu notebook. Cell sẽ tìm `Lab10/qa_model.py` hoặc clone repo vào `/content/NLP`.
3. Chạy cell cài dependency; xác nhận `torch.cuda.is_available()` là `True` trước khi FULL.
4. Hoàn thành `prediction.md`, đọc `student_implementation_guide.md`, rồi tự triển khai các TODO alignment và metric trong `qa_model.py`.
5. Chạy `tokenizer_alignment.ipynb`; sau đó chọn `MODE="QUICK"`, bật `RUN_TRAINING=True` để smoke test; chỉ khi QUICK qua mới chuyển `MODE="FULL"`.
6. Tải `Lab10/outputs/`, notebook outputs và PNG/CSV về máy hoặc đồng bộ lên Drive. Checkpoints bị ignore để tránh commit file lớn.

Nếu notebook mở trực tiếp từ GitHub nhưng runtime báo thiếu thư mục, cần clone repo trong runtime; mở notebook từ trang GitHub không tự tạo checkout trong `/content`.

## Configuration and reproducibility

`QAConfig` ở `qa_model.py` là cấu hình chung. Đề xuất: model/tokenizer `distilbert-base-uncased`, seed 42, batch size 8, learning rate `3e-5`, 2 epochs, AdamW, max length mặc định 384, stride mặc định 128 và FP16 khi có CUDA. Thí nghiệm context length dùng stride chung 64 cho 128/256/384 vì stride phải nhỏ hơn max length; giữ các yếu tố còn lại cố định và khởi tạo lại từ cùng pretrained model cho mỗi setting.

Các số trên là **proposed configuration**, chưa phải actual run. Actual config, hardware, thời gian, số examples, split IDs và kết quả chỉ được ghi sau khi chạy thành công. Test không dùng chọn checkpoint; validation EM/F1 chọn best checkpoint. So sánh context length trên held-out split là báo cáo thí nghiệm, không dùng chọn cấu hình để tuyên bố test độc lập lần nữa.

## Execution order and mapping to W10.pdf

1. `calculations.md`: mục 6–8, 12, 19 và learning check.
2. `prediction.md`: mục 9; hoàn thành trước khi xem baseline/experiment outputs.
3. `tokenizer_alignment.ipynb`: mục 10, 11, 13, 15, 16 — tokenizer inputs, offsets, overflow windows và test cases.
4. `qa_model.py`: baseline mục 21, QA model start/end forward, train utilities, TODO interfaces cho alignment, decoding và EM/F1.
5. `qa_finetuning.ipynb`: mục 14, 17, 20, 22, 23, 24, 25; QUICK trước, FULL sau.
6. `document_qa.py`: mục 27 — câu hỏi và context/tệp text được cung cấp trực tiếp, không có retrieval/RAG.
7. `error_analysis.md`, `reflection.md`: sinh viên tự hoàn thiện từ output thật.

## Student-owned TODOs — must finish before training

`qa_model.py` cố ý để `NotImplementedError` tại:

- `character_to_token_span`, `prepare_train_features`, `prepare_validation_features`, `window_answer_coverage`.
- `decode_qa_predictions`.
- `normalize_answer`, `exact_match_score`, `token_f1_score`, `score_one_example`, `evaluate_predictions`.
- `classify_span_errors`.

`student_implementation_guide.md` giải thích nguyên lý, công thức, ví dụ, thứ tự làm và checklist edge cases mà không thay sinh viên viết các hàm này.

Notebook không bật training mặc định. Chỉ bật sau khi alignment/metric đã tự cài đặt và test do sinh viên hoàn tất. `tests/test_qa_framework.py` kiểm thử phần boilerplate đã triển khai; các alignment/metric tests được ghi rõ **SKIPPED — student TODO**, không tính là PASS.

## Outputs

Sau run thật, notebook ghi:

- `results.csv` gồm experiment/model/setting/split, số mẫu, max length, stride, window count, answer coverage, training steps, EM/F1, training time và seed. EM/F1 theo thang 0–100 như SQuAD evaluator.
- `outputs/training_history.csv`, run config, model predictions và error candidate evidence.
- `learning_curves.png`, `em_f1_analysis.png`, `answer_length_analysis.png`.
- `error_analysis.md` chỉ chứa candidate/evidence; cột Explanation để trống.

Nếu một run chưa hoàn thành, giữ trạng thái NOT_RUN/INCOMPLETE, không thêm dòng metrics giả. Khi tự hoàn thiện module metrics, giữ thang 0–100 nhất quán với quy ước SQuAD evaluator.

## Local checks performed for this handoff

- `python -m unittest discover -s tests -v`: **13 passed, 32 skipped, 0 failed**. Skips are the student-owned alignment, feature, decoding, metric and error-classification cases.
- `python -m py_compile qa_model.py document_qa.py tests/*.py`: passed.
- Both notebooks: JSON valid and all code cells parse successfully; notebook cells were **not executed**.
- No SQuAD download, pretrained training, QA result CSV or result plot was produced in this local check. CUDA is unavailable in the local runtime.

## AI usage and remaining work

- **Generated:** project boilerplate, dataset split scaffold, keyword-overlap baseline, model loading/forward/train utilities, notebook structure, plots/export scaffold, and tests for non-restricted utilities.
- **Modified:** split policy made reproducible and article-disjoint; one stride (64) is proposed for controlled context-length runs.
- **Verified:** framework tests and syntax checks listed above. No SQuAD model training or experiment metrics are claimed yet.
- **Student must complete:** calculations, predictions, character/token alignment, span decoding, QA EM/F1, error explanations, reflection and learning checks.

## Known limitations

- The public dataset split has no labeled test split in this setup; evaluation uses a reproducible article-disjoint partition of its original validation split. Report it as a held-out evaluation partition, not an official hidden test.
- The model cannot provide reliable no-answer behavior from SQuAD v1.1. SQuAD 2.0 is optional and excluded from core.
- Training and all metric plots remain NOT_RUN until student TODOs are implemented and a Colab/local runtime completes the experiments.

## Current readiness

- Framework, split checks, result upsert, checkpoint/resume utilities, gradient accumulation and experiment notebook scaffolds are implemented.
- QUICK and FULL are **not runnable yet**: training remains gated until student alignment, span decoding and EM/F1 implementations pass their tests.
- No SQuAD download, pretrained fine-tuning, full experiment, metric CSV or result plot is claimed as completed by this checkout.
- Colab T4 execution is prepared but not verified from this local environment. Confirm `torch.cuda.is_available()` in Colab, run QUICK first, and use FULL only after all gates pass.
