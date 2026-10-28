# LAB 06 — Attention Mechanism

## Folder layout

Notebook, source code, W6.pdf, README and student-written documents stay in the Lab6 root. All generated images, result CSV/NPZ/TXT files and model checkpoints are collected under `artifacts/`. The folder also contains the conventional names `results.csv`, `attention_heatmap.png` (trained Experiment 4 attention), and `confusion_matrix.png` (Attention model).

```text
Lab6/
├── W6.pdf
├── README.md
├── calculations.md
├── prediction.md
├── attention_numpy.py
├── self_attention.ipynb
├── error_analysis.md
├── reflection.md
└── artifacts/
    ├── results.csv
    ├── attention_heatmap.png
    ├── confusion_matrix.png
    └── experiment/application outputs and checkpoints
```

## Experiment 1 — Attention distribution

Experiment 1 minh họa dot-product attention trên câu mẫu trong W6.pdf. Query là vector giả lập, còn key vectors được tạo bằng NumPy với seed `42`; đây không phải embedding học được từ ngôn ngữ hay sentiment model.

### Files

- `attention_numpy.py`: stable softmax và dot-product attention chưa scaling.
- `self_attention.ipynb`: tạo dữ liệu tái lập, kiểm tra phép tính, hiển thị bảng và vẽ phân phối attention.
- `artifacts/experiment1_attention_distribution.csv`: token, score và attention weight được tính khi chạy notebook.
- `artifacts/experiment1_attention_distribution.png`: biểu đồ được lưu khi chạy notebook.

### Dependencies

- Python 3
- NumPy, pandas, Matplotlib, scikit-learn, PyTorch, Hugging Face `datasets`
- Jupyter Notebook để mở/chạy notebook

### Run

Từ thư mục `Lab6`:

```bash
python -m pip install numpy matplotlib jupyter
python -m pip install numpy pandas matplotlib scikit-learn torch datasets jupyter
jupyter notebook self_attention.ipynb
```

Chạy các cell từ trên xuống dưới. E1–E3 dùng seed `42`; seed của split IMDb LAB 05 là `2026`.

### Công thức

Experiment 1 dùng **unscaled** dot product theo yêu cầu của mục 18:

```text
scores = K @ q
weights = stable_softmax(scores)
```

Không chia scores cho `sqrt(d_k)` trong thí nghiệm này. Các query/key vectors được mô phỏng ngẫu nhiên và chỉ dùng để quan sát phép tính; không diễn giải weights như độ quan trọng ngữ nghĩa của từ.

### AI assistance statement

- Tool: OpenAI Codex.
- Purpose: tạo helper NumPy, notebook, biểu đồ và kiểm thử Experiment 1.
- Generated: code cho stable softmax, dot-product attention và khung thực nghiệm.
- Modified: giữ phép tính không scaling, dùng đúng token/câu và seed được chỉ định trong đề.
- Verified: chạy notebook, kiểm tra shapes, tính hữu hạn, chuẩn hóa softmax, tính thứ tự score-weight, tái lập và artifacts CSV/PNG.

## Experiment 2 — Query variation

Experiment 2 tiếp tục trong các cell cuối của `self_attention.ipynb`. Nó giữ nguyên 10 token và key matrix đã tạo ở Experiment 1, dùng Query A hiện có và so sánh với Query B `[0.25, -0.5, 0.5, 1.0]`. Hai query có cùng L2 norm; chỉ query thay đổi. Cả hai dùng cùng hàm `dot_product_attention`, softmax, temperature `1.0` và không scaling.

Chạy notebook từ đầu đến cuối như hướng dẫn ở trên. Kết quả được lưu riêng, không ghi đè outputs Experiment 1:

- `artifacts/experiment2_query_comparison.csv`
- `artifacts/experiment2_query_comparison.png`
- `artifacts/experiment2_query_heatmap.png`

Notebook kiểm tra cùng sequence/keys, normalization, reproducibility, CSV/PNG và regression outputs của Experiment 1. Query labels `Sentiment` và `Topic` chỉ là tên minh họa cho vectors giả lập; chúng không phải representation đã học.

### AI assistance statement — Experiment 2

- Tool: OpenAI Codex.
- Purpose: tích hợp query variation, bảng, biểu đồ và kiểm thử regression.
- Generated: các cell Experiment 2 trong notebook và hướng dẫn chạy.
- Modified: tái sử dụng query/keys/hàm từ Experiment 1; không đổi seed, keys, scaling hay temperature.
- Verified: thực thi notebook, kiểm tra weights, khác biệt phân phối, tính tái lập và artifacts; xác nhận Experiment 1 không đổi.

## Experiment 3 — Attention visualization

Mục này dùng hai câu ở trang 13 của W6.pdf. Mỗi token có synthetic embedding kích thước 4; cùng một token dùng cùng vector trong cả hai câu. Các ma trận WQ, WK, WV kích thước 4×4 được tạo bằng NumPy với seed `42` và dùng chung cho hai câu.

Công thức được thực hiện trong `attention_numpy.self_attention`:

```text
Q = X @ WQ
K = X @ WK
V = X @ WV
scores = (Q @ K.T) / sqrt(d_k)
weights = stable_softmax(scores, axis=-1)
context = weights @ V
```

`stable_softmax` hiện thực subtract-max và hỗ trợ ma trận theo hàng. Notebook gọi helper này, hiển thị hai heatmap, top-3 attention targets cho `cat`, `adopted`, `sleeping`, cùng biểu đồ so sánh. Chạy notebook từ đầu đến cuối bằng lệnh ở trên.

Experiment 3 tạo các artifacts riêng, không ghi đè Experiment 1–2:

- `artifacts/experiment3_attention_heatmap_sentence1.png`
- `artifacts/experiment3_attention_heatmap_sentence2.png`
- `artifacts/experiment3_selected_tokens_comparison.png`
- `artifacts/experiment3_attention_matrix_sentence1.csv`
- `artifacts/experiment3_attention_matrix_sentence2.csv`
- `artifacts/experiment3_selected_token_analysis.csv`

Kiểm chứng bằng cách xem mỗi ma trận có shape 8×8, tổng từng hàng xấp xỉ 1, CSV có đúng nhãn hàng/cột và các PNG được lưu. Embeddings/projections là dữ liệu ngẫu nhiên giả lập; heatmap chỉ mô tả phân phối attention theo các vector đó, không phải bằng chứng về quan hệ ngữ pháp.

### AI assistance statement — Experiment 3

- Tool: OpenAI Codex.
- Purpose: triển khai NumPy self-attention helper, visualization, CSV và tests.
- Generated: `self_attention` trong `attention_numpy.py`, helper artifacts, các cell notebook và hướng dẫn này.
- Modified: dùng cùng seed, shared token embeddings và cùng WQ/WK/WV cho cả hai câu; không sửa Experiment 1–2.
- Verified: thực thi notebook, kiểm tra dimensions, row-wise softmax, reproducibility, selected top-3, files và regression outputs trước đó.

## Experiment 4 — LSTM context vs attention context

Mục tiêu là so sánh representation của hai sentiment classifiers trên cùng IMDb splits. Dữ liệu được nạp bằng Hugging Face `imdb` (chỉ split có nhãn): IMDb train gốc được chia stratified 80/20 thành train/validation với `random_state=2026`; IMDb test gốc được giữ làm test. Đây là chính cách chia trong `Lab5/sequence_classification.ipynb` (20,000/5,000/25,000 samples). Vocabulary tối đa 20,000 mục (gồm PAD/UNK) được xây từ train con; tokenizer, HTML cleanup, `max_length=100`, right truncation, padding, embedding dimension 50, batch 64, Adam `0.001`, 3 epochs và BCEWithLogitsLoss theo LAB 05.

Run này dùng bản IMDb đã có trong Hugging Face cache. Khi chưa có cache, `datasets` cần kết nối Hugging Face Hub để tải dữ liệu.

`experiment4_models.py` chạy hai model độc lập:

- **LSTM:** Embedding → một-layer unidirectional LSTM → final hidden state → binary classifier. Packed sequences bỏ qua padding.
- **Attention:** Embedding → tự cài Q/K/V scaled dot-product attention bằng phép toán PyTorch → mask padded keys → masked mean pooling → classifier. Không dùng positional encoding hoặc MultiheadAttention.

Seed khởi tạo/training là `42`; split seed được giữ `2026` để khớp LAB 05. Validation binary F1 chọn checkpoint; test chỉ được dùng cho báo cáo cuối. Training time đo bằng `perf_counter()` từ trước epoch đầu đến sau epoch cuối và gồm thời gian validation trong các epoch. Cấu hình, phiên bản thư viện, checkpoint, histories, metrics, confusion matrices, representation inspection và test report được lưu riêng với tiền tố `experiment4_`; không ghi đè E1–E3 hay checkpoint LAB 05.

Artifacts chính:

- `artifacts/experiment4_results.csv`, `artifacts/experiment4_training_history.csv`, `artifacts/experiment4_config.json`
- `artifacts/experiment4_training_loss.png`, `artifacts/experiment4_validation_f1.png`
- `artifacts/experiment4_confusion_matrix_lstm.png`, `artifacts/experiment4_confusion_matrix_attention.png`
- `artifacts/experiment4_representation_inspection.csv`, `artifacts/experiment4_attention_matrix.csv`
- `artifacts/experiment4_trained_attention_heatmap.png`, `artifacts/experiment4_test_report.txt`
- `artifacts/checkpoints/experiment4_lstm.pt`, `artifacts/checkpoints/experiment4_attention.pt`

### Kết quả lần chạy đã lưu

Test metrics là kết quả thực tế từ 25.000 review; F1 là binary F1 với positive label `1`. Thời gian đo trên CPU bằng cùng phạm vi (training epochs và validation), còn checkpoint được chọn bằng Validation F1.

| Model | Accuracy | F1 | Training time (s) | Parameters |
|---|---:|---:|---:|---:|
| LSTM | 0.7152 | 0.7449 | 57.7517 | 1,029,761 |
| Attention | 0.7899 | 0.7929 | 7.8320 | 1,007,551 |

Kết quả của E4 lấy từ các model train thực tế. Attention is not guaranteed to outperform LSTM. Representation mechanisms differ even when both models solve the same sentiment classification task. Attention heatmap là phân phối tính được từ model này; không phải bằng chứng về linguistic reasoning. Mô hình attention không positional encoding kết hợp masked mean pooling không trực tiếp biểu diễn thứ tự token như LSTM.

### AI Assistance Statement — Experiment 4

- Tool: OpenAI Codex.
- Purpose: hỗ trợ implementation, debugging, testing, visualization và tài liệu.
- Generated: module E4, notebook cells và artifacts code.
- Modified: theo tokenizer/split/vocabulary/truncation và cấu hình LAB 05; tách output E4 khỏi thí nghiệm trước.
- Verified: notebook training, test metrics, mask/padding/gradient checks, file checks và regression E1–E3.

## Application 26 — Context-aware sentiment classification

Ứng dụng dùng trực tiếp checkpoint Attention đã train ở Experiment 4: `artifacts/checkpoints/experiment4_attention.pt`. Không tạo model ngẫu nhiên, không fine-tune và không xây vocabulary mới. Checkpoint chứa token-to-ID vocabulary; script kiểm tra PAD/UNK, cấu hình E4 và label mapping trước khi inference. Dataset huấn luyện là IMDb labeled splits; labels đã xác minh: `0 = Negative`, `1 = Positive`.

Chạy từ thư mục `Lab6`:

```bash
python application26_context_sentiment.py
```

Preprocessing được tái sử dụng từ `experiment4_models.py`: lowercase, HTML unescape/tag cleanup, regex tokenizer `[a-z0-9]+(?:'[a-z]+)?`, mapping OOV sang `UNK_ID`, giữ 100 token đầu và right padding bằng `PAD_ID`. Attention chạy trên model ở `eval()`/`inference_mode`; xác suất positive lấy từ sigmoid của binary logit, xác suất negative là `1 - P(positive)`.

Ba input giữ nguyên theo W6.pdf:

1. `I like this movie.` — expected Positive.
2. `I do not like this movie.` — expected Negative.
3. `I thought this movie would be bad, but it was actually excellent.` — expected Positive.

Expected labels là tham chiếu theo đề, không phải ground truth IMDb. Kết quả lần chạy đã lưu:

| Sentence | Expected | Predicted | P(Positive) | P(Negative) | Correct? |
|---|---|---|---:|---:|---|
| 1 | Positive | Negative | 0.126321 | 0.873679 | No |
| 2 | Negative | Negative | 0.006060 | 0.993940 | Yes |
| 3 | Positive | Positive | 0.990302 | 0.009698 | Yes |

Correct/incorrect examples lấy từ ba câu bắt buộc: câu 2 là correct theo expected label, câu 1 là incorrect. Không cần dùng thêm review test. CSV lưu cả tokens, token IDs, OOV/UNK, truncation, xác suất và shape. Sinh viên tự đối chiếu câu, nhãn tham chiếu và ma trận khi viết error analysis.

### Output files

- `artifacts/application26_predictions.csv`: predictions và preprocessing evidence cho ba câu.
- `artifacts/application26_attention_sentence1.png`, `artifacts/application26_attention_sentence2.png`, `artifacts/application26_attention_sentence3.png`: ba heatmap riêng từ trained model; trục X là key, trục Y là query, padding không hiển thị, color scale dùng chung [0, 1].
- `artifacts/application26_attention_matrices.npz`: attention matrices và token labels gốc.
- `artifacts/application26_attention_received.csv`: `mean(A[:, token_index])` trên các query positions hợp lệ. Đây chỉ là phép tổng hợp weights, không phải thước đo nhân quả.
- `artifacts/application26_selected_examples.csv`, `artifacts/application26_selected_correct_attention.png`, `artifacts/application26_selected_incorrect_attention.png`: evidence cho một mẫu đúng và một mẫu sai.
- `artifacts/application26_test_report.txt`: kiểm tra runtime.

Heatmaps là attention weights được lấy trực tiếp từ forward pass của checkpoint E4, không phải synthetic attention của Experiment 3. Weights mô tả phân phối attention, không chứng minh mô hình hiểu phủ định, tương phản hay linguistic reasoning. Output probabilities cũng chưa được đánh giá calibration. Attention-only architecture của E4 không có positional encoding.

### AI Assistance Statement — Application 26

- Tool: OpenAI Codex.
- Purpose: tái sử dụng checkpoint, viết inference/visualization, verification và documentation.
- Generated: `application26_context_sentiment.py`, output tables/figures và phần README này.
- Modified: không thay architecture/checkpoint/vocabulary; E1–E4 artifacts được snapshot và so sánh byte-for-byte.
- Verified: checkpoint strict-load, sentence inference, probability/mask/normalization checks, selected correct/error evidence, repeat inference và output validation; 16 runtime checks PASS.


## Organizing and refreshing result artifacts

After running the experiment/application scripts, all generated files are written under `artifacts/`. Rebuild the combined summary CSV from the experiment-specific CSVs with:

```bash
python build_results_summary.py
```

This creates `artifacts/results.csv`; source CSVs are retained for traceability.
