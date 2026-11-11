# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

Triển khai theo `W8.pdf` (16 trang). Các phần code dùng Python/PyTorch và Hugging Face; bài tính tay, prediction, giải thích lý thuyết, interpretation, error hypotheses và reflection để sinh viên tự làm.

## Mục tiêu và cấu trúc

Thực hành tokenizer/subword/special tokens/mask; so sánh initialization, freezing, lượng training data, sequence length và domain shift; lưu metrics và bằng chứng thực tế để người học tự phân tích.

```text
Lab8/
├── W8.pdf
├── README.md
├── calculations.md
├── prediction.md
├── tokenization.ipynb
├── pretrained_classification.py
├── fine_tuning.ipynb
├── results.csv
├── data_efficiency.png
├── confusion_matrix.png
├── confidence_analysis.png
├── error_analysis.md
├── reflection.md
├── requirements.txt
├── test_lab08.py
├── completion_report.md
└── outputs/
    ├── unit_tests.log
    ├── verification.json
    ├── smoke/
    └── quick/ (full/ được tạo khi chạy full)
```

Tên `Lab8` giữ theo repo hiện có; các tên deliverable trong W8.pdf được giữ nguyên. Hình chỉ được tạo khi có dữ liệu đã đo. `outputs/<mode>/` chứa config/môi trường, source-ID manifests, status, log, history, test predictions, error candidates và checkpoint từng run. Checkpoint `best_model/` không đưa vào Git do dung lượng; chúng vẫn được lưu local để inference/resume. Lưu thêm chúng vào Drive nếu cần giữ sau khi Colab hết phiên.

## Cài môi trường

```bash
cd /path/to/NLP/Lab8
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Trong Jupyter/Colab dùng `%pip install -r requirements.txt`; trong Bash dùng `python -m pip`, không dùng `%pip`. Các version đã kiểm tra nằm trong `outputs/<mode>/environment.json`; requirements khai báo khoảng tương thích, không ép bản PyTorch CPU vào Colab GPU.

## Dataset, nhãn và split

- [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb): chỉ dùng 25.000 train và 25.000 test có nhãn; không dùng unsupervised. `0=Negative, 1=Positive`.
- IMDb được nạp từ hai file Parquet `train` và `test` của repository Hugging Face tại commit đã pin trong code; cách này tránh gọi `imdb.py` cũ đã bị gỡ khỏi nhánh dataset hiện tại. Vì vậy không dùng split `unsupervised`.
- Train chính thức được chia stratified 80/20, seed **42**: full là **20.000 train / 5.000 validation / 25.000 test**. Đây là cấu hình LAB08, không lấy nguyên split seed 2026 của LAB07.
- Quick/smoke lấy mẫu stratified từ ba split gốc, giữ source ID để kiểm tra không có một sample index thuộc nhiều split. Validation/test cố định giữa mọi run cùng config.
- Audit exact duplicate text được ghi riêng. Nếu dữ liệu chính thức có văn bản trùng giữa splits, pipeline cảnh báo và không tự thay split/xóa review. Xem `duplicate_text_audit.csv` trước khi khẳng định dữ liệu không bị contamination.
- [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity): Domain B chỉ dùng test. Loader đọc file Parquet từ bản sao được công bố là duplicate của dataset này để tránh builder script cũ. Nhãn gốc `['1','2']`; encoded index **0 = negative, 1 = positive**. Không train hoặc chọn checkpoint bằng Yelp.
- Nếu không tải được IMDb và không có cache, phần training dừng với lỗi; không thay bằng dữ liệu giả. Có thể cung cấp `--imdb-dir /path` chứa `train.csv`, `test.csv` thật với `text,label`.
- Nếu Yelp không khả dụng, Bài 18 được đánh dấu BLOCKED; có thể dùng `--domain-b-file domain_b.csv` có `text,label` (0/1 đúng nghĩa sentiment). Không giả định mọi dataset có hai nhãn đều tương thích.

## Model và classification head

Model mặc định là `distilbert/distilbert-base-uncased`; có thể chọn `--model-name bert-base-uncased`. Tokenizer luôn đi cùng model. Revision thực tế được lưu trong từng `complete.json` và `results.csv`; `--revision` cho phép pin commit khi chạy lại.

- Pretrained: `AutoModelForSequenceClassification.from_pretrained`.
- Scratch: cùng `AutoConfig` và tokenizer nhưng tạo weights bằng `from_config`. Không tải pretrained weights cho model scratch.
- “Scratch” ở đây chỉ nói về model weights; vocabulary/subword tokenizer được dùng chung với pretrained baseline theo thiết kế kiểm soát.
- DistilBERT classifier thực tế lấy vector token đầu, qua `pre_classifier` Linear → ReLU → dropout → `classifier` Linear. Notebook/CLI in các modules thật, không giả định head chỉ có một Linear. Xem [implementation chính thức](https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py).
- Base checkpoint chưa được fine-tune sentiment; task head được khởi tạo ngẫu nhiên. Inference trước training chỉ là kiểm tra API, không phải đánh giá classifier sentiment đã học.
- Frozen mode khóa `model.base_model`, giữ encoder ở eval mode khi train head; fine-tuning cập nhật encoder và head. Programmatic assertions và test backward kiểm tra điều này.
- [Dynamic padding](https://huggingface.co/docs/transformers/data_collators) pad tới sequence dài nhất trong batch; tokenizer truncation bên phải và max length bao gồm special tokens. Mọi input truyền attention mask.

## Chạy tokenization và inference

Mở `tokenization.ipynb` và Run All. Bài 10 có token IDs thật, `convert_ids_to_tokens()`, raw `word_ids()` và mapping theo whitespace words. Content pieces bao gồm dấu câu, còn `##` đếm riêng continuation pieces; không đánh đồng các số này.

Kiểm tra head chưa train (CLI sẽ in cảnh báo):

```bash
python pretrained_classification.py --inference "I enjoyed this movie." "The ending was disappointing."
```

Inference với checkpoint đã fine-tune:

```bash
python pretrained_classification.py --checkpoint outputs/quick/RUN_ID/best_model \
  --inference "I did not enjoy this movie."
```

Thay RUN_ID bằng đường dẫn trong `outputs/quick/*/complete.json`; không lấy checkpoint random-head làm classifier cuối.

## Chạy experiments

CLI quick mặc định: 2.000 train / 500 validation / 500 test, 2 epochs. Full: 20.000 / 5.000 / 25.000, 3 epochs. Smoke: 32 / 16 / 16, 1 epoch; chỉ kiểm tra kỹ thuật.

```bash
python pretrained_classification.py --mode quick
python pretrained_classification.py --mode full --scratch-data-curve
```

Lệnh của quick pilot CPU được dùng trong lượt triển khai:

```bash
python pretrained_classification.py --mode quick --train-size 400 \
  --validation-size 200 --test-size 200 --epochs 2 --batch-size 8 \
  --gradient-accumulation-steps 1 --max-length 128 --scratch-data-curve --local-files-only
```

`--local-files-only` chỉ dùng khi checkpoint đã được cache. Khi cần tải lần đầu, bỏ cờ này. Nếu chủ động chạy offline và đã có cache, có thể đặt `HF_HUB_OFFLINE=1` trước lệnh. `fine_tuning.ipynb` lưu cùng cấu hình pilot; cell config chỉ bật local-files-only khi có `HF_HUB_OFFLINE=1`, còn Colab mặc định tự tải.

Có thể chọn experiment và giữ nguyên các giá trị config khác:

```bash
python pretrained_classification.py --mode quick --experiments 14 15
python pretrained_classification.py --mode quick --experiments 16 17
python pretrained_classification.py --mode quick --experiments 18 19
```

- **14:** scratch vs pretrained, cùng architecture/tokenizer, data, LR, epochs và max length.
- **15:** pretrained frozen vs fine-tuning từ cùng base weights/seed, không train tiếp từ model trước.
- **16:** nested 10/25/50/100%; pretrained luôn được hỗ trợ, thêm scratch curve bằng `--scratch-data-curve`.
- **17:** pretrained fine-tuning length 64/128/256, còn lại giữ nguyên. CUDA peak reset trước mỗi run; CPU peak để null nếu chưa đo.
- **18:** cùng checkpoint train trên IMDb, đánh giá IMDb và Yelp; delta = B − A.
- **19–21:** reference pretrained/full-train-fraction với max length cấu hình chính, final test metrics, report, confusion matrix, error candidates và confidence groups.

Run trùng config/data/environment được tái sử dụng từ `complete.json` và checkpoint, không retrain. Các experiment dùng chung baseline có cùng run ID khi điều kiện thực sự giống nhau. Mỗi run khác configuration có hash riêng. CSV upsert theo experiment/run/split/domain, giữ kết quả cũ và không tạo duplicate row. Run lỗi chỉ ghi status/exception; không điền metrics giả.

## Google Colab GPU T4

### Dùng T4 trong VS Code local

Extension chính thức **Google Colab** đã có trong VS Code Server hiện tại (v0.9.7). Để kết nối notebook đang mở ở local tới máy Colab:

1. Mở `Lab8/colab_t4_full_run.ipynb` trong VS Code và chọn **Select Kernel → Colab → New Colab Server**. Đăng nhập Google khi được yêu cầu.
2. Chọn cấu hình GPU **T4** nếu Colab đang cấp phát được. Tên menu có thể hiện là **GPU (T4)**. T4 là máy từ xa của Colab; code notebook vẫn được sửa/lưu ở workspace local.
3. Trong Explorer, nhấp phải `pretrained_classification.py` và `requirements.txt` → **Upload to Colab**. Upload hai file này lên `/content`; không cần chuyển các checkpoint CPU cũ.
4. Chạy các cell trong launcher từ trên xuống. Notebook kiểm tra file nguồn, xác nhận đúng GPU T4, cài requirements, gắn Drive và chạy full CLI. Nếu muốn kiểm tra GPU thủ công, dùng:

```python
!nvidia-smi
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

Nếu thiếu GPU hoặc T4 không có trong danh sách, Colab chưa cấp GPU cho phiên đó; kiểm tra lại **Select Kernel → Colab → New Colab Server** hoặc thử lại sau. Việc cài extension không tự tạo runtime và không bảo đảm T4 luôn sẵn có.

`fine_tuning.ipynb` đang giữ output của quick pilot **400/200/200, 2 epochs**. Nếu bước tự gắn Drive trong launcher không mở được xác thực, gọi **Colab: Mount Google Drive to Server** trong Command Palette của VS Code.

```python
from pathlib import Path
OUTPUT_DIR = Path("/content/drive/MyDrive/NLP/Lab8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
!python -u /content/pretrained_classification.py --mode full --device cuda --scratch-data-curve --output-dir /content/drive/MyDrive/NLP/Lab8
```

Full mode dùng toàn bộ IMDb có nhãn: 20.000 train, 5.000 validation, 25.000 test, 3 epochs. Kết quả và checkpoint được ghi vào Drive để có thể tái sử dụng run đã hoàn thành nếu runtime bị ngắt. Khi lệnh chạy xong, tải `outputs/`, CSV, báo cáo và hình về local qua Explorer (**Download from Colab**) nếu cần lưu vào repo. Colab có giới hạn phiên và GPU không được bảo đảm luôn sẵn có.

### Dùng Colab trên web

Chọn **Runtime → Change runtime type → T4 GPU**, rồi chạy các cell:

```python
!git clone https://github.com/NguyenDucThang-tb/NLP.git
%cd /content/NLP/Lab8
%pip install -r requirements.txt
```

Sau khi `Lab8` đã có trên GitHub, chạy `tokenization.ipynb` rồi `fine_tuning.ipynb`, hoặc CLI:

```python
!python pretrained_classification.py --mode quick --device cuda
!python pretrained_classification.py --mode full --device cuda --scratch-data-curve
```

FP16 chỉ bật với CUDA; AdamW, dynamic padding, gradient accumulation có trong config. Mặc định batch=8, accumulation=2; có thể giảm batch và tăng accumulation khi thiếu VRAM, nhưng phải ghi sự thay đổi trong so sánh. Full có nhiều run và có thể mất nhiều giờ. GPU/T4 chưa được gọi là đã kiểm chứng nếu chỉ chạy trên CPU.

Lưu toàn bộ Lab8/outputs cùng các PNG/CSV sau khi chạy; checkpoint trong Colab sẽ mất khi runtime bị xóa nếu chưa sao lưu.

## Reproducibility và protocol

Nguồn số liệu: `outputs/<mode>/config.json`, `environment.json`, các manifests và `*/complete.json`. Mỗi record ghi model/revision, seed, fraction, sample counts, max length, batch, LR, epochs, parameter counts, thiết bị, thời gian và config hash.

- Seed=42 cho Python, NumPy, PyTorch và stratified split.
- AdamW; LR mặc định 2e-5, weight decay 0.01; max_length=128; gradient clipping=1.0.
- Checkpoint chọn bằng **validation macro-F1**; loss tại checkpoint đó không được gọi là minimum validation loss.
- Bảng/curve so sánh trong notebook dùng **validation**. Test riêng ở Bài 18/19. Reference config được định trước, không chọn bằng test F1.
- `training_time_seconds` chỉ cộng thời gian các epoch training (forward/backward/optimizer); không gồm download/tokenization/validation/checkpoint I/O. `validation_time_seconds` và wall time lưu riêng trong complete.json.
- CUDA peak là allocated memory trong training/validation, không phải toàn bộ GPU utilization. CPU peak không đo thì để null.
- Thời gian một lần chạy chịu ảnh hưởng tải máy, I/O và warm-up; fixed budget chung không chứng minh mỗi initialization đã được tối ưu LR riêng.
- Thời gian đo chỉ thuộc downstream training; chi phí tạo pretrained checkpoint trước đó không được cộng vào số này. Initialization của head cũng cần xem cùng seed/API khi đánh giá độ ổn định; frozen/full dùng cùng pretrained checkpoint và head được kiểm tra bằng test.
- Confidence: High ≥0.90; Medium ≥0.70 và <0.90; Low <0.70. Nhóm rỗng có accuracy null. Không giả định softmax đã được calibrated.
- Mọi kết quả quick/smoke có mode và sample count thật. Chưa chạy full không được dùng các số này thay cho full.

## Kiểm thử

```bash
python -m unittest discover -s . -p test_lab08.py -v
python pretrained_classification.py --mode smoke --experiments 14 --max-length 64 \
  --batch-size 4 --gradient-accumulation-steps 1
```

Tests dùng fixture nhỏ chỉ để kiểm tra logic; fixtures không được ghi thành kết quả IMDb. Notebook validation, executions và kiểm tra artifacts được ghi trong `outputs/verification.json`. Trạng thái theo từng bài nằm trong `completion_report.md`.

## Results overview

<!-- MEASURED_RESULTS_START -->
Quick pilot CPU: 400 train / 200 validation / 200 IMDb test, 2 epochs; seed=42. Full và GPU T4: NOT_RUN.

|   experiment_id | initialization   | train_mode   |   train_samples |   max_length | split      | domain        |   accuracy |   f1_macro |   training_time_seconds |
|----------------:|:-----------------|:-------------|----------------:|-------------:|:-----------|:--------------|-----------:|-----------:|------------------------:|
|              14 | scratch          | fine_tuning  |             400 |          128 | validation | IMDb          |     0.5000 |     0.3333 |                158.3619 |
|              14 | pretrained       | fine_tuning  |             400 |          128 | validation | IMDb          |     0.7950 |     0.7947 |                165.1205 |
|              15 | pretrained       | frozen       |             400 |          128 | validation | IMDb          |     0.6400 |     0.5929 |                 40.5257 |
|              15 | pretrained       | fine_tuning  |             400 |          128 | validation | IMDb          |     0.7950 |     0.7947 |                165.1205 |
|              18 | pretrained       | fine_tuning  |             400 |          128 | test       | IMDb          |     0.8000 |     0.7995 |                165.1205 |
|              18 | pretrained       | fine_tuning  |             400 |          128 | test       | Yelp Polarity |     0.8500 |     0.8493 |                165.1205 |
|              19 | pretrained       | fine_tuning  |             400 |          128 | test       | IMDb          |     0.8000 |     0.7995 |                165.1205 |

14 unit tests và cả hai notebook đã execute; xem `outputs/verification.json`. Các số trên là quick, không dùng thay kết quả full.
<!-- MEASURED_RESULTS_END -->

## AI Usage

| Mục | Khai báo kỹ thuật |
|---|---|
| AI-assisted code | OpenAI Codex: module pipeline, notebooks, tests và visualization |
| What was generated | Tokenization/classification/training helpers, controlled experiments, artifact export, templates tự làm |
| What was modified | Nội dung LAB08 mới trong Lab8; giữ code LAB01–07 |
| What was verified | Chỉ những lần chạy trong logs/verification.json; không khẳng định người học đã tự review/verify |

## Giới hạn và Student TODO

MLM calculation, prediction trước experiment, giải thích pretraining/fine-tuning, interpretation của từng bảng, error type/hypothesis, sáu câu hỏi Bài 28, learning check và reflection đều dành cho sinh viên tự viết. Chạy code không thay thế các nội dung này. Xem báo cáo trạng thái để phân biệt IMPLEMENTED, TESTED, EXECUTED, GENERATED, BLOCKED và STUDENT TODO.
