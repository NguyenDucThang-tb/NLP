# Hướng dẫn các phần sinh viên tự triển khai — LAB 10

Tài liệu này giải thích đầu vào, đầu ra, công thức và thứ tự làm cho các hàm còn `NotImplementedError` trong `qa_model.py`. Hãy tự viết hàm và bật test sau khi tự xác nhận expected output. Character offsets dùng khoảng **[start, end)**; chỉ số token start/end là inclusive.

## 1. Character span sang token span

Hàm: `character_to_token_span(offset_mapping, sequence_ids, answer_start, answer_text)`.

- Character gold span là `[answer_start, answer_start + len(answer_text))`.
- Chỉ xét token có `sequence_id == 1` (context), offset hợp lệ và có giao với gold span.
- Với token offset `[a,b)`, hai khoảng giao nhau khi `a < gold_end` và `b > gold_start`.
- Trước khi trả token span, kiểm tra answer có nằm trọn trong vùng ký tự của context window: `gold_start >= first_context_offset_start` và `gold_end <= last_context_offset_end`. Nếu không, answer đã bị cắt ở một cạnh hoặc nằm ngoài window; trả `None`.
- Sau đó, các token context giao với answer phải phủ được cả hai đầu mút: token giao đầu bắt đầu không muộn hơn `gold_start`, token giao cuối kết thúc không sớm hơn `gold_end`. Trả index token giao đầu/cuối theo dạng inclusive.
- Trả `None` nếu không có token giao hoặc answer không được window phủ trọn. Khi tạo train feature, caller dùng quy ước no-answer-window (thường CLS cho cả hai nhãn). Không dùng token câu hỏi hoặc special token làm nhãn trừ CLS theo quy ước đó.

Ví dụ riêng cho một window có context offsets `[(100, 104), (105, 109), (110, 114)]` và sequence IDs `[1, 1, 1]`:

| Trường hợp | Gold character span | Kết quả alignment |
|---|---:|---|
| Nằm trọn trong window | `[105,109)` | `(1,1)` |
| Bị cắt ở đầu window | `[98,104)` | `None` |
| Bị cắt ở cuối window | `[112,118)` | `None` |
| Nằm ngoài window | `[20,23)` | `None` |

Kiểm tra thêm: ranh giới answer là end-exclusive; offset `(0,0)` của special/padding không phải context; answer lặp lại phải dùng `answer_start` đã gán nhãn để chọn occurrence. Chỉ kiểm tra overlap sẽ nhầm hai trường hợp bị cắt một phần thành span hợp lệ.

## 2. Tạo train features và sliding windows

Hàm: `prepare_train_features(tokenizer, examples, max_length, stride)`.

1. Tokenize theo cặp `(question, context)`, `truncation="only_second"`, bật overflow và offset mapping.
2. `overflow_to_sample_mapping[j]` cho biết feature thứ `j` sinh từ example ban đầu nào.
3. Lấy `sequence_ids(j)` để chỉ nhận diện token thuộc context.
4. Dùng answer character span của example tương ứng để tìm token span trong từng window.
5. Nếu window chứa trọn answer, gắn start/end token positions. Nếu không, dùng quy ước no-answer-window nhất quán (thường CLS cho cả hai đầu trong SQuAD v1.1).
6. Trả các model inputs cùng `start_positions`, `end_positions`, và metadata `overflow_to_sample_mapping` cần để đo coverage.

Ví dụ hình dung: một context dài sinh ra ba windows; nếu answer chỉ xuất hiện trọn trong window 2 thì window 2 nhận span label, hai window còn lại nhận nhãn no-answer-window. Cùng một câu gốc có thể vì vậy tạo nhiều training features.

## 3. Validation features và answer coverage

Hàm `prepare_validation_features()` tokenize cùng cách nhưng không dùng gold span để tạo nhãn. Mỗi feature cần giữ `example_id` hoặc sample mapping, `offset_mapping`, và sequence IDs/đủ thông tin để biết token nào là context. Không bỏ các example không có answer trong window.

Hàm `window_answer_coverage(example, feature)` trả lời câu hỏi: window này có chứa **toàn bộ** gold character span không? So sánh span answer với vùng offset context của feature, đồng thời xác nhận các token giao span thuộc context. Coverage của tập dữ liệu là số example có ít nhất một window chứa đủ answer chia cho tổng số example.

## 4. Decode start/end logits

Hàm: `decode_qa_predictions(examples, features, start_logits, end_logits, max_answer_length=30)`.

1. Gom feature theo example gốc bằng mapping/ID.
2. Tạo danh sách vị trí hợp lệ chỉ từ context tokens; loại question, special và padding.
3. Xét cặp `(start, end)` hợp lệ với `start <= end` và độ dài token không vượt `max_answer_length`.
4. Chấm cặp bằng `start_logit + end_logit`; giữ cặp điểm cao nhất qua mọi windows của cùng example.
5. Đổi token span thành character span qua offset mapping và cắt `context[start_char:end_char]` từ context gốc.
6. Nếu không có candidate hợp lệ, trả prediction rỗng cùng offset rỗng/None theo schema đã chọn.

Inference chỉ được dùng question, context, offsets và logits. Không đọc `answers`, `answer_start` hoặc bất kỳ gold metadata nào để chọn span dự đoán.

## 5. SQuAD normalization, Exact Match và token F1

Chuẩn hóa câu trả lời trước khi so sánh: lowercase, bỏ punctuation, bỏ mạo từ tiếng Anh `a`, `an`, `the`, rồi chuẩn hóa whitespace.

- Exact Match cho một reference: `1` nếu hai chuỗi đã chuẩn hóa giống nhau, nếu không là `0`.
- Token F1: token hóa chuỗi đã chuẩn hóa; tính overlap theo **multiset** để từ lặp không được đếm vượt số lần xuất hiện; `precision = overlap / #prediction tokens`, `recall = overlap / #gold tokens`, `F1 = 2PR/(P+R)`.
- Nhiều gold answers: lấy điểm lớn nhất trong các reference được chấp nhận.
- Prediction và reference đều rỗng: xử lý theo quy ước SQuAD; prediction rỗng nhưng gold không rỗng có F1/EM bằng 0.
- Dataset EM/F1 là trung bình theo example, nhân 100 để khớp thang điểm SQuAD.

Ví dụ: prediction `"the blue car"`, reference `"blue car"`: sau normalization cùng thành `"blue car"`, nên EM là 1 và F1 là 100 trên thang phần trăm. Prediction `"blue vehicle"` với reference `"blue car"` có overlap một token: token precision và recall đều `1/2`, F1 là `0.5` (50 trên thang phần trăm).

## 6. Error evidence theo character offsets

`classify_span_errors()` nhận `answers` dạng `{"text": [...], "answer_start": [...]}` và một prediction. So sánh predicted `[start_char,end_char)` với tất cả gold spans ở tọa độ context gốc, không đối chiếu token index giữa windows. Trả về bốn trường `wrong_start`, `wrong_end`, `wrong_span` (boolean) và `error_type` (nhãn mô tả kiểu sai biên). Nếu dự đoán khớp một trong nhiều gold spans, không đánh dấu sai span. Đây chỉ là phân loại hình học của span; không suy ra nguyên nhân ngôn ngữ.

## 7. Test cases cần tự xác nhận

- Alignment: answer một token/nhiều token, ranh giới giữa subword, offset end-exclusive, answer ngoài window, bị truncation, có mặt trọn trong window, text lặp, nhiều windows, question/special/padding.
- Decode: chỉ context, start không vượt end, max answer length, answer từ window có score cao nhất, context substring đúng offsets, không dùng gold trong inference.
- Metrics: exact, partial overlap, no overlap, case, punctuation, articles, whitespace, repeated tokens, empty strings, nhiều references.
- Split: không giao article title giữa train/validation/held-out; IDs không lặp trong mỗi split.

`tests/test_qa_student.py` để các test ở trạng thái skipped cho đến khi sinh viên tự ghi expected values đã kiểm chứng. Không đổi expected output chỉ để làm test qua.
