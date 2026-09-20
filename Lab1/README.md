# LAB 01 - From Text Processing to Search

## Cách chạy trên Linux/macOS

Mở terminal tại thư mục `Lab1`:

```bash
cd /home/nguyenducthang/NLP/Lab1
python3 implementation.py
# Mở và chạy experiments.ipynb bằng Jupyter/VS Code
```

Nếu máy bạn dùng lệnh `python` cho Python 3 thì có thể thay `python3` bằng
`python`. Trên Windows, dùng PowerShell và chạy tương tự với đường dẫn Windows.

`implementation.py` là phần tự cài TF-IDF lõi, không gọi `TfidfVectorizer`.
`experiments.ipynb` chạy pipeline trên file JSONL 30K, thực hiện Part D
(retrieval experiment) và Part F (evaluation/error-analysis scaffold).

## Thứ tự làm bài bắt buộc

1. Tự viết `calculations.md` và `prediction.md` trước khi chạy code.
2. Chạy unit tests của `implementation.py`.
3. Mở `experiments.ipynb`, chạy các cell Part D và đọc `part_D_results.csv`.
4. Dựa trên số liệu thật để viết `reflection.md` và error analysis.

## Lưu ý về AI

Không dùng phần giải thích dưới đây như câu trả lời nộp nguyên xi. Các mục
calculation, prediction, numerical-results explanation, error analysis và
reflection phải thể hiện suy nghĩ của chính bạn. Nếu giữ mã nguồn được hỗ trợ
bởi AI, hãy khai báo rõ contribution trong `reflection.md` theo yêu cầu của đề.

## Các file bài làm

`calculations.md` là tên file chính theo đề. `calculation.md` là bản mẫu tương
thích được giữ lại để tránh mất nội dung cũ; khi nộp, hoàn thiện và nộp một bản
duy nhất, ưu tiên `calculations.md`.

## Dataset

Mỗi dòng JSON có các trường `text`, `timestamp`, `url`. Trong kết quả, số thứ
tự `document_id` là chỉ số dòng bắt đầu từ 0 trong file JSONL.
