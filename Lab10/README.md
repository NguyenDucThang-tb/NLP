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
├── tests/test_qa_student.py
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
4. Đọc `student_implementation_guide.md` để hiểu alignment, window labels, span decoding và EM/F1 đã được triển khai trong `qa_model.py`.
5. Mở `qa_finetuning.ipynb`, chọn **Runtime → Run all**. Notebook được cấu hình `MODE="FULL"`, `RUN_TRAINING=True`, và sẽ chạy unit tests trước. FULL dừng ngay nếu không phát hiện CUDA. Nếu muốn kiểm tra nhanh trước, đổi `MODE="QUICK"`; QUICK dùng 64/16/16 examples và một epoch, còn FULL dùng tối đa 10.000/1.000/1.000 examples và hai epochs.
6. Tải `Lab10/outputs/`, notebook outputs và PNG/CSV về máy hoặc đồng bộ lên Drive. Checkpoints bị ignore để tránh commit file lớn.

Nếu notebook mở trực tiếp từ GitHub nhưng runtime báo thiếu thư mục, cần clone repo trong runtime; mở notebook từ trang GitHub không tự tạo checkout trong `/content`.

## Configuration and reproducibility

`QAConfig` ở `qa_model.py` là cấu hình chung. Đề xuất: model/tokenizer `distilbert-base-uncased`, seed 42, batch size 8, learning rate `3e-5`, 2 epochs, AdamW, max length mặc định 384, stride mục tiêu 64 và FP16 khi có CUDA. Notebook kiểm tra chiều dài câu hỏi của các split đang dùng; nếu stride 64 không vừa context capacity ở max length 128, nó tự hạ stride chung và in giá trị thực tế. Cả ba setting dùng cùng stride và khởi tạo lại từ cùng pretrained model.

Các số trên là **proposed configuration**, chưa phải actual run. Actual config, hardware, thời gian, số examples, split IDs và kết quả chỉ được ghi sau khi chạy thành công. Test không dùng chọn checkpoint; validation EM/F1 chọn best checkpoint. So sánh context length trên held-out split là báo cáo thí nghiệm, không dùng chọn cấu hình để tuyên bố test độc lập lần nữa.

## Execution order and mapping to W10.pdf

1. `calculations.md`: mục 6–8, 12, 19 và learning check.
2. `prediction.md`: mục 9; hoàn thành trước khi xem baseline/experiment outputs.
3. `tokenizer_alignment.ipynb`: mục 10, 11, 13, 15, 16 — tokenizer inputs, offsets, overflow windows và test cases.
4. `qa_model.py`: baseline mục 21, QA model, training utilities, alignment, feature preparation, span decoding, EM/F1 và phân loại sai biên.
5. `qa_finetuning.ipynb`: mục 14, 17, 20, 22, 23, 24, 25; QUICK trước, FULL sau.
6. `document_qa.py`: mục 27 — câu hỏi và context/tệp text được cung cấp trực tiếp, không có retrieval/RAG.
7. `error_analysis.md`, `reflection.md`: sinh viên tự hoàn thiện từ output thật.

## Implemented algorithms and test gate

Các hàm alignment, train/validation feature preparation, answer coverage, span decoding, chuẩn hóa SQuAD, Exact Match, token F1, aggregation theo ID và phân loại span errors đã được triển khai trong `qa_model.py`. `tests/test_qa_student.py` chứa expected cases hoạt động, không còn test TODO bị skip.

Trước khi training, `qa_finetuning.ipynb` chạy toàn bộ unit tests. Các cờ alignment/metrics/error-analysis chỉ được mở nếu suite thành công và không có test skip. Notebook mặc định chạy FULL sau khi người học chọn GPU T4; QUICK là lựa chọn kiểm tra nhanh.

FULL giữ `RUN_ID="full_seed42"` ổn định và bật tự resume. Nếu Colab ngắt, mở lại runtime, chọn T4 và chạy notebook lại: checkpoint có `trainer_state.pt` sẽ resume; context length chưa có trạng thái sẽ bắt đầu từ pretrained weights. Muốn chạy mới từ đầu, đổi `RUN_ID` sang tên mới hoặc đặt `RESUME_EXISTING=False`.

Các triển khai được tạo với AI hỗ trợ. Hãy đọc guide, rà lại công thức và hiểu output tests trước khi chạy; khai báo AI contribution trong reflection. Không có kết quả training nào được tạo trong bước kiểm thử code này.

## Outputs

Sau run thật, notebook ghi:

- `results.csv` gồm experiment/model/setting/split, số mẫu, max length, stride, window count, answer coverage, training steps, EM/F1, training time và seed. EM/F1 theo thang 0–100 như SQuAD evaluator.
- `outputs/training_history.csv`, run config, model predictions và error candidate evidence.
- `learning_curves.png`, `em_f1_analysis.png`, `answer_length_analysis.png`.
- `error_analysis.md` chỉ chứa candidate/evidence; cột Explanation để trống.

Nếu một run chưa hoàn thành, giữ trạng thái NOT_RUN/INCOMPLETE, không thêm dòng metrics giả. Khi tự hoàn thiện module metrics, giữ thang 0–100 nhất quán với quy ước SQuAD evaluator.

## Local verification status

- Unit suite: **47 passed, 0 failed, 0 skipped** after implementing the QA algorithms, checkpoint resume handling and model-input compatibility.
- `qa_model.py`, `document_qa.py` and tests pass Python syntax compilation.
- No pretrained SQuAD fine-tuning, formal result CSV or experiment plots have been generated; the local runtime has no CUDA.
- Colab T4 execution has not been verified from this local environment.

## AI usage and remaining work

- **Generated with AI assistance:** project boilerplate, split scaffold, keyword baseline, training utilities, notebook setup, alignment/feature preparation, span decoding, SQuAD metrics, span-error categories and tests.
- **Modified:** split policy made reproducible and article-disjoint; one stride (64) is proposed for controlled context-length runs.
- **Verified:** all 45 unit tests and Python syntax checks pass. No pretrained SQuAD model training or experiment metrics are claimed yet.
- **Student must complete:** inspect/understand the AI-assisted algorithms, run QUICK/FULL on Colab, and write evidence-based interpretation, error explanations, reflection and learning checks.

## Known limitations

- The public dataset split has no labeled test split in this setup; evaluation uses a reproducible article-disjoint partition of its original validation split. Report it as a held-out evaluation partition, not an official hidden test.
- The model cannot provide reliable no-answer behavior from SQuAD v1.1. SQuAD 2.0 is optional and excluded from core.
- Training and experiment metric plots remain NOT_RUN until QUICK/FULL actually run in a suitable runtime.

## Current readiness

- Framework, split checks, result upsert, checkpoint/resume utilities, gradient accumulation and experiment notebook scaffolds are implemented.
- Code and test gates are configured for Colab FULL; the notebook runs tests before training and requires CUDA in FULL mode.
- No pretrained fine-tuning, full experiment, metric CSV or result plot is claimed as completed by this checkout.
- GPU execution is **NOT_VERIFIED**. Confirm CUDA/T4 in Colab, finish QUICK first, and run FULL only after reviewing its actual outputs.
