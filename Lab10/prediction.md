# Prediction before experiment — LAB 10

> Hoàn thành trước khi chạy baseline hoặc training. Không xem kết quả rồi mới ghi prediction.

## Prediction 1 — Lexical overlap and distractors

Prediction: Model có khả năng chọn nhầm câu trả lời khi context chứa nhiều từ trùng với question nhưng chỉ có một đoạn thực sự chứa đáp án đúng.

Reason: Model có thể chú ý quá mức đến những token có độ tương đồng từ vựng cao với question. Tuy nhiên, pretrained Transformer có khả năng khai thác thông tin ngữ cảnh nên dự kiến sẽ ít mắc lỗi hơn baseline dựa trên keyword overlap.

Confidence: 85%.

## Prediction 2 — Answer length

Prediction: Answer span càng dài thì việc dự đoán chính xác start và end position càng khó.

Reason: Với answer dài, model có thể xác định đúng một boundary nhưng dự đoán sai boundary còn lại, dẫn đến câu trả lời bị thiếu hoặc thừa token. Vì vậy, EM có khả năng giảm mạnh hơn F1 khi độ dài answer tăng.

Confidence: 85%.

## Prediction 3 — Irrelevant context

Prediction: Khi context chứa nhiều thông tin không liên quan, EM và F1 của model có khả năng giảm.

Reason: Thông tin không liên quan làm tăng số lượng candidate spans, khiến model khó xác định đoạn chứa câu trả lời chính xác. Ngoài ra, context dài có thể khiến answer bị truncation hoặc không nằm trọn trong một sliding window.

Confidence: 90%.

## Prediction 4 — Sentence order

Prediction: Việc thay đổi thứ tự các câu trong context có thể ảnh hưởng đến kết quả dự đoán, mặc dù answer vẫn được giữ nguyên.

Reason: Transformer sử dụng positional information và self-attention để xây dựng contextual representations. Khi thứ tự câu thay đổi, vị trí token và mối quan hệ ngữ cảnh cũng thay đổi. Tuy nhiên, nếu các câu độc lập và answer vẫn nằm trọn trong context window thì mức độ ảnh hưởng có thể nhỏ.

Confidence: 75%.
