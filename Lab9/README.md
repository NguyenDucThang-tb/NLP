# LAB 09 — Named Entity Recognition and Information Extraction

Chủ đề: chuyển từ sequence classification sang token classification bằng CoNLL-2003 và BIO tags.

## Cài đặt

Tạo môi trường riêng rồi cài dependency:

```bash
cd /home/nguyenducthang/NLP/Lab9
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Trong Colab, mở notebook từ repository. Nếu PyTorch đã được cài cho GPU, không cài đè torch; cài các gói còn thiếu theo `requirements.txt`.

## Dataset và model

- Dataset: [CoNLL-2003 trên Hugging Face](https://huggingface.co/datasets/tner/conll2003), giữ nguyên train/validation/test. Loader ưu tiên JSONL và `label.json` của cùng dataset để tương thích với Datasets mới không chạy dataset scripts.
- Dataset card mô tả bốn entity types `PER`, `ORG`, `LOC`, `MISC` và nêu các điều khoản bản quyền của dữ liệu Reuters; người dùng cần tuân thủ điều khoản nguồn.
- Base model: `distilbert/distilbert-base-cased`; tokenizer fast.
- Seed: 42. Các nhãn được lấy từ metadata, không tự gán thứ tự label ID.

## Chạy

Thứ tự đề nghị:

1. Hoàn thành `prediction.md` trước khi chạy Experiment 1 hoặc baseline chính thức.
2. Mở và chạy `tokenizer_alignment.ipynb` từ trên xuống dưới để xem word/subword alignment và Experiment 1.
3. Chạy baseline:

   ```bash
   ./.venv/bin/python baseline_ner.py
   ```

   Có thể dùng dữ liệu CoNLL-2003 local chuẩn token-per-line bằng `--data-dir PATH`.
4. Mở `ner_finetuning.ipynb`. Sau khi prediction đã hoàn thành, đặt `RUN_FULL_TRAINING = True`; notebook giữ validation cho chọn checkpoint và dùng test một lần ở bước đánh giá cuối.
5. Chạy Information Extraction sau khi có checkpoint:

   ```bash
   ./.venv/bin/python information_extraction.py \
     --checkpoint outputs/checkpoints/best \
     --text "Barack Obama visited Paris."
   ```

6. Chạy unit tests:

   ```bash
   ./.venv/bin/python -m unittest -v test_lab09.py
   ```

## Cấu hình và artifacts

Fine-tuning mặc định: DistilBERT cased, max length 256, batch size 8, 3 epochs, AdamW, learning rate `2e-5`, weight decay `0.01`, seed 42. Runtime, package versions, GPU/CPU và dataset split được ghi vào `outputs/environment.json`; training arguments và logs sẽ được ghi dưới `outputs/` khi thật sự chạy.

- `results.csv`: model metrics; chỉ có dòng cho mô hình đã chạy.
- `entity_metrics.csv`: per-type strict entity scores.
- `confusion_matrix.png`: token-level BIO confusion matrix, có ghi rõ loại metric.
- `entity_length_analysis.png`: strict exact-span F1 theo số original words trong entity.
- `error_analysis.md`: evidence của baseline sau khi baseline chạy; phần Transformer chỉ xuất hiện sau khi có test predictions.
- `context_analysis.md`: same surface form với gold types khác nhau từ test split.
- `ie_gold_template.csv`: 30 câu chọn từ CoNLL-2003 test, gold spans lấy trực tiếp từ annotation dataset.
- `ie_evaluation.csv`: chỉ được tạo metrics sau khi có checkpoint và chạy dự đoán trên 30 câu.
- `outputs/`: predictions, checkpoint và metadata; checkpoint không được commit tự động.

## Phần cần sinh viên tự làm

`calculations.md`, `prediction.md`, `reflection.md`, các nhận xét cuối thí nghiệm, giải thích lỗi và learning check được để TODO. Hãy tự hoàn thành theo W9.pdf và chính các output đã chạy. Nếu `prediction.md` còn TODO thì notebook sẽ bỏ qua training chính thức.

## Khai báo AI hỗ trợ

- **Generated:** boilerplate Python cho data loading, tokenizer alignment, baseline, entity metrics, notebook cells, plots và CLI Information Extraction.
- **Modified:** code được điều chỉnh để dùng JSONL/label metadata khi Datasets v4 từ chối thực thi dataset script; thêm kiểm tra BIO, truncation và word-level reconstruction.
- **Verified:** unit tests và các phần dataset/tokenizer/baseline chỉ được ghi là verified khi lệnh chạy tương ứng hoàn tất. Chưa chạy không được xem là kết quả.

## Trạng thái thực nghiệm

Xem `experiment_report.md` để biết số liệu nào lấy từ lần chạy thật và phần nào chưa chạy. Baseline hiện tại đã được chạy lại sau khi hoàn thành `prediction.md` và có trạng thái `EXECUTED`. Output lưu trong `tokenizer_alignment.ipynb` có từ lần exploratory trước đó; hãy chạy lại notebook để refresh Experiment 1 theo đúng thứ tự. Không coi CPU smoke test là kết quả fine-tuning Transformer.
